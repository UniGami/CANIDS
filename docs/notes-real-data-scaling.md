# Notes: Scaling Branch 1 Training to Real SynCAN

Not a PLAN.md step write-up like the numbered docs — this is an investigation
into why the pipeline built and tested against the small synthetic dataset
(Steps 4-9) doesn't scale to the real SynCAN dataset (added later, see
`canids/data/syncan.py`) as-is, what it would actually cost, and what to do
about it. Captured here so the reasoning survives past the chat session that
did the measuring.

## Where training actually happens

The gradient-descent loop is [`gru_seq2seq.py`](../src/canids/models/gru_seq2seq.py)'s
`train()` function:

- `for start in range(0, n, batch_size)` — the batch loop
- `pred = model(xb)` — forward pass through the GRU
- `loss.backward()` — backprop
- `optimizer.step()` — the weight update
- once per epoch, a single **unbatched** forward pass over the *entire*
  validation set computes `val_loss` (its own, smaller version of the memory
  problem below)

It's called from `scripts/run_detector.py`'s `main()`:
`history = train(model, X_train, y_train, X_val, y_val, epochs=args.epochs, seed=args.seed)`,
fed by `X_train, y_train = make_forecast_windows(...)`, which builds the
*entire* window tensor in memory, up front, before training starts.

## The real dataset, measured

Inspecting the actual zip (not the README's stated schema — see
`canids/data/syncan.py`'s docstring for the two real mismatches found:
column names without the `_of_ID` suffix, and Time recorded in milliseconds
not seconds):

- 4 training files, each **7,417,430 rows, spanning 15,526s (~4.31 hr)** of
  simulated driving time. Concatenated per the SynCAN README's own
  recommendation: **~29.7M rows / ~17.25 hours** of driving time.
- 10 real CAN IDs, 20 signals total, so `registry.vector_size == 40` (vs.
  the synthetic dataset's 14).
- At the default 10ms grid step and 50-tick sequence length, that's ~6.2M
  forecast windows total (~5.0M train / ~1.2M val after the 80/20 split).

## Compute time: ~25 hours, and that's the SMALLER problem

Benchmarked directly on the development machine used to build this (CPU-only
PyTorch, no CUDA at the time of measurement, 8 cores): **2,778
windows/second** (forward + backward + optimizer step, at real SynCAN's
shapes — vector_size=40, hidden=64, batch=64).

```
50 epochs × 4,968,192 train windows ÷ 2,778 windows/sec ≈ 89,400s ≈ 24.8 hours
```

— if compute were the only constraint. It isn't.

## Memory: the actual blocker

`make_forecast_windows` (via `data/windowing.py`'s `drop_windows_with_nan`,
`windows[valid]`) **copies** every sliding window into one big array before
training starts. Windows overlap heavily — each tick appears in ~51
different windows (`sequence_length + 1`), once for every window that
includes it — so this copy duplicates almost every number in the dataset
**~51 times over**.

Concretely, for one training file: the raw joint-state vector is only
~500MB (1.55M ticks × 40 signals × 8 bytes). The windowed copy is
`1,552,560 windows × 51 ticks × 40 signals × 8 bytes ≈ 25GB` — and
500MB × 51 ≈ 25.5GB confirms the duplication-factor reasoning exactly, it
isn't a coincidence.

`train()` then converts that array to a `torch.float32` tensor
(`torch.tensor(X_train, dtype=torch.float32)`); because the source is
`float64`, this can't reuse the buffer and forces **another** ~12GB copy,
alongside the first, not replacing it. And that's for **one of four**
training files — concatenating all four (as the README recommends, and as
`canids/data/syncan.py`'s `concatenate_with_time_offset` supports) multiplies
this past **100GB**, far beyond a typical machine's RAM. The realistic
outcome isn't "slow," it's a crash (`MemoryError`) or the OS silently
swapping to disk, which is so much slower it looks like a hang.

**The fix, not yet implemented**: replace the "build every window as one
array" approach with a `torch.utils.data.Dataset`/`DataLoader` that keeps
the ~500MB joint vector in memory exactly once, and slices out each
window's 50 ticks on demand inside `__getitem__`. A batch of 64 then costs
about 0.5MB instead of the current 25GB+ all at once. This also fixes the
validation-loss computation's smaller version of the same problem. This is
a prerequisite for training on real data at any serious scale — not
optional, and not yet done.

## Scaling plan: every issue, every lever, measured where possible

Numbers below are **measured** on this development machine (CPU-only
PyTorch, 8 cores) unless explicitly marked *estimated* — this section
distinguishes the two throughout, so nothing here should be mistaken for a
benchmark that wasn't actually run.

### Issues to address, ranked by whether they block a run at all

| # | Issue | Severity | Fix |
|---|---|---|---|
| 1 | Full window materialization (`windows[valid]`) copies the whole windowed dataset into one array — ~25GB/file, ~100GB+ for all four concatenated | **Blocking** — crashes or swaps to disk | Streaming batch gather (below) |
| 2 | Validation loss computed as a single **unbatched** forward pass over the entire validation set (`gru_seq2seq.py` `train()`, `model(X_val_t)`) | **Blocking** — same failure mode as #1, smaller scale (~10GB for one file) | Batch it, same fix as #1 |
| 3 | `TRAINING_EPOCHS=50` was tuned against the synthetic dataset's ~9,500 windows/epoch; real data is ~500x denser per epoch | Inefficiency, not a blocker | Early stopping on validation loss instead of a fixed count |
| 4 | Default `batch_size=64` under-uses both CPU and (eventually) GPU parallelism | Inefficiency | Raise batch size; see measurements below |
| 5 | No CUDA-enabled PyTorch installed, despite real GPU hardware present | Missed opportunity, not a blocker | Reinstall `torch` with a CUDA build (see GPU section) |
| 6 | `calibration.cusum_statistic`'s CUSUM recursion is an explicit Python loop, run once per signal | Minor — measured ~16s for a 1.2M-tick validation set across 20 signals | Not worth fixing yet; would matter more if Step 13's `sensitivity_sweep` (5 percentiles) or repeated evaluation runs make it recur often |
| 7 | Whether training needs **all four** train files at all, or whether one already saturates what an unsupervised forecaster can learn from this data, hasn't been checked | Scope/methodology question, not an engineering bug | Train on one file first, inspect the validation-loss curve, only add more if it's still improving meaningfully at the end |
| 8 | One-time costs — `pd.read_csv` on a 7.4M-row file, `align_to_grid`'s forward-fill over ~1.55M ticks × 20 signals — haven't been benchmarked at full scale | Unknown, but structurally these are vectorized pandas/numpy operations over tens of millions of elements, which routinely run in low single-digit seconds; not expected to be a meaningful fraction of an hours-long training run | Spot-check once a full real CSV is actually on disk; not a priority |

### The streaming fix, specifically (issues #1 and #2)

The naive fix — a `torch.utils.data.Dataset` that returns one window per
`__getitem__` call — would remove the memory problem but risks *reintroducing*
a speed problem: a Python-level function call per training example (millions
of them) adds real overhead that the current single-tensor-slice approach
doesn't have. The right shape for this fix is a **batch-level gather**: keep
the ~500MB joint vector and a small array of valid start-tick indices in
memory, and for each batch, gather that batch's windows with one vectorized
NumPy fancy-index operation (`joint_vector[starts[:, None] + np.arange(seq_len + 1)]`)
— a few MB per batch, not gigabytes, and no per-sample Python overhead. This
is simple enough to write as a plain generator function feeding the existing
batch loop, without necessarily needing `DataLoader`'s multiprocessing
machinery — the data itself is already in memory, so there's no I/O to
overlap with compute. **Expected effect on speed: neutral** — this fix's job
is making the run possible at all, not making it faster. The speedups below
are independent of it.

### Measured throughput (this machine, CPU, GRU hidden_size=64, vector_size=40)

| Configuration | Throughput | vs. baseline |
|---|---|---|
| Train, batch=64 (current default) | 2,684 windows/sec | 1.0x |
| Train, batch=256 | 4,048 windows/sec | 1.51x |
| Train, batch=1024 | 6,253 windows/sec | 2.33x |
| Eval (no_grad), batch=1024 | 18,278 windows/sec | — |
| Eval (no_grad), batch=8192 | 21,459 windows/sec | (diminishing returns past ~1024) |

### Stacked estimate, all four training files concatenated (~5.0M train / ~1.24M val windows)

| Stage | What changed | Time/epoch | Epochs | Total | Cumulative speedup |
|---|---|---|---|---|---|
| Baseline | batch=64, fixed 50 epochs, unbatched val (~68s/epoch estimated at the batch=1024 eval rate as a stand-in) | ~1,919s | 50 | **~26.7 hours** | 1.0x |
| + streaming fix | (memory only — run now actually *completes* instead of crashing) | ~1,919s | 50 | ~26.7 hours | 1.0x |
| + batch_size 1024 | train 6,253/s, val 18,278/s | ~862.5s | 50 | ~12.0 hours | 2.23x |
| + early stopping (~10 epochs) | validated via the val-loss curve, not hardcoded | ~862.5s | 10 | **~2.40 hours** | **11.1x** |
| + GPU *(estimated, not measured)* | see caveat below | — | 10 | **~18-48 minutes** | ~33-89x |
| + single file instead of 4 *(methodology choice, issue #7)* | 1/4 the windows | — | 10 | **~4.5-12 minutes** (with GPU) / ~36 min (CPU only) | up to ~355x |

**The software-only result (batch size + early stopping, no GPU, no reduced
data) is the headline finding: ~26.7 hours → ~2.4 hours, an ~11x reduction,
using nothing but measurements already taken on this machine.** GPU and
reduced-data-volume are real additional levers on top of that, but the first
11x doesn't require new hardware or a methodology decision — it's just
fixing two config values (`batch_size`, `epochs`/early-stopping) once the
streaming fix makes the run possible.

**GPU caveat, stated plainly**: the 3-8x range used above (narrower than
the 5-15x quoted earlier in this doc, now that it's being used in a load-bearing
calculation) is an *estimate*, not a measurement — this machine's PyTorch
install is still CPU-only. GRU/RNN workloads specifically parallelize less
well on GPU than CNNs or Transformers, because each of the 50 timesteps in a
sequence has to be computed one after another regardless of hardware — the
GPU wins by processing more *sequences in the same batch* simultaneously,
not by speeding up one sequence's 50 sequential steps. That means the GPU
win is real but batch-size-dependent, and should be **measured**, not
trusted from this estimate, once a CUDA build is installed.

### Optional extra lever: window striding

Not included in the stacked table above because it trades away data rather
than removing waste. Training on every Nth window instead of every window
(adjacent windows already share 49 of 50 ticks, so the loss in diversity is
small) divides both the per-epoch time and the total proportionally to N —
stride=2 halves everything above again, stride=5 divides by 5, and so on.
Worth reaching for only if the stacked plan above still isn't fast enough
for a given deadline, not as a default.

### Recommended order of implementation

1. [x] **Streaming batch-gather fix** — implemented. `data/windowing.py` gained
   two primitives: `valid_forecast_ticks()` (finds every valid target tick
   in one O(n_ticks × vector_size) scan of the joint vector itself, using
   the fact that NaN only ever forms a single leading prefix — no per-window
   scan, no window-sized copy) and `gather_forecast_batch()` (builds exactly
   one batch's (X, y) via a single vectorized NumPy gather, a few MB, not
   gigabytes). `models/gru_seq2seq.py`'s new `train_streaming()` uses both;
   `train()` is untouched, so the synthetic-dataset tests and existing call
   sites keep working exactly as before.
2. [x] **Batch validation loss** — implemented in the same
   `train_streaming()`, via `_batched_eval_loss()`, replacing `train()`'s
   single unbatched `model(X_val_t)` call.
3. [x] **Raise `batch_size`** — `scripts/run_detector.py` now defaults
   `--batch-size` to 256 (was hardcoded to `config.BATCH_SIZE == 64`);
   `--batch-size 1024` is available for larger real-data runs.
4. [x] **Early stopping** — `train_streaming()` takes
   `early_stopping_patience`/`min_delta`; `run_detector.py` defaults
   `--early-stopping-patience` to 5 (`0` disables it, running the full
   `--epochs`).
5. [ ] Train on one file first (issue #7) — a methodology decision for
   whoever runs the real training, not a code change; still open.
6. [ ] CUDA reinstall — still open; a real, separate action (package
   reinstall, several GB download), deliberately last since items 1-4 alone
   already deliver the largest, best-grounded speedup.

Verified end-to-end against both the synthetic dataset and a real SynCAN
slice (`scripts/prepare_syncan_data.py` output) after implementing 1-4: the
full pipeline (train → calibrate → correlation graph → detect → attribute)
runs correctly through the new streaming path, including reproducing the
suppression-detection result (1711/1712 ground-truth ticks caught) from
before this change, confirming the optimization didn't alter detection
behavior — only how the windows get built. `predict_streaming()` (the
inference-side counterpart, used for both validation residuals and test-set
detection) and `naive.residuals_streaming()` were added alongside so no
call site in `run_detector.py` still needs a fully pre-built window array.

### Function-by-function breakdown of what changed

- **`data/windowing.py`, `valid_forecast_ticks(joint_vector, sequence_length)`**
  — replaces "build every window, then check each one for NaN" with one
  direct fact: `data/grid.py` guarantees NaN only ever occupies a single
  leading prefix of `joint_vector` (the warm-up before every signal's first
  transmission), never reappearing after. So this function finds the first
  fully-valid tick with one `np.isnan(joint_vector).any(axis=1)` scan
  (touches ~500MB once, no window-shaped copy), then every target tick from
  there to the end is automatically valid — `np.arange`, no scanning. Empty
  input, an all-NaN array, and data shorter than `sequence_length` are all
  handled explicitly rather than left to produce a confusing downstream
  error.
- **`data/windowing.py`, `gather_forecast_batch(joint_vector, registry, target_ticks, sequence_length)`**
  — the streaming fix's actual data-fetching primitive. Turns a small array
  of target tick indices into that batch's `(X, y)` with one vectorized
  NumPy fancy-index gather (`joint_vector[starts[:, None] + offsets[None, :]]`),
  costing only that batch's size — for batch=1024 at real SynCAN's shapes,
  a few MB, not the 25GB+ a full materialization would cost. Same
  `(X, y)` contract as `make_forecast_windows_with_ticks`, just one batch
  at a time instead of the whole dataset at once.
- **`models/gru_seq2seq.py`, `train_streaming(model, registry, train_joint, val_joint, ...)`**
  — same optimization loop shape as `train()` (Adam, MSE, one
  `zero_grad`/`backward`/`step` per batch), but sources every batch from
  `gather_forecast_batch` on demand instead of slicing a pre-built tensor.
  Shuffles by permuting *tick indices* (`np.random.default_rng(seed).permutation(train_ticks)`)
  rather than permuting a pre-built array's rows, since there's no array to
  permute anymore. Validation loss is computed by the new
  `_batched_eval_loss()` helper — same batching, `model.eval()` +
  `torch.no_grad()`, replacing `train()`'s single giant forward call.
  `early_stopping_patience`/`min_delta` are additive, off-by-default-shaped
  parameters (`None` reproduces `train()`'s always-run-`epochs` behavior
  exactly) so nothing about `train()`'s existing contract changed.
- **`models/gru_seq2seq.py`, `predict_streaming(model, registry, joint_vector, ticks, ...)`**
  — the inference-side counterpart: batches through `gather_forecast_batch`
  under `torch.no_grad()`, returns `(y, pred)` as two concatenated
  `(len(ticks), n_signals)` arrays. Concatenating is cheap here specifically
  *because* it's only the small per-window output being concatenated, not
  the large per-window input — the thing that was actually expensive to
  materialize was always the `sequence_length`-wide context, not the
  single-tick prediction.
- **`models/naive.py`, `residuals_streaming(joint_vector, registry, ticks)`**
  — naive persistence never needed windowing in the first place (its
  "prediction" is just the value one tick earlier), so this is a direct,
  unbatched two-line gather (`joint_vector[ticks]` vs. `joint_vector[ticks - 1]`)
  — added purely so no caller of the GRU's streaming path needs to fall back
  to the old pre-built-array naive functions just to run the Step 7
  confidence-gating comparison.
- **`scripts/run_detector.py`** — training, validation-residual computation,
  and test-set detection all switched from `make_forecast_windows`/`make_forecast_windows_with_ticks`
  + `train`/`predict`/`residuals` to `valid_forecast_ticks` +
  `train_streaming`/`predict_streaming`/`residuals_streaming`. Added
  `--batch-size` (default 256) and `--early-stopping-patience` (default 5)
  CLI flags; the training-summary print now reports how many epochs
  actually ran vs. the `--epochs` cap, since early stopping can end a run
  early.

### Testing notes

Every new primitive is checked against the exhaustive method it replaces on
the same data, not just "does it run": `valid_forecast_ticks` against
`make_forecast_windows_with_ticks`'s tick indices (both on the tiny
hand-crafted fixture and on a realistically-shaped synthetic dataset with
several irregularly-transmitting IDs, so the NaN-prefix assumption is
checked at more than one scale), and `gather_forecast_batch` against the
same function's `(X, y)` output, including for an arbitrary partial subset
of ticks (the actual shape of how it's called — one mini-batch at a time).

The important one is `test_train_streaming_matches_train_with_one_batch_per_epoch`:
it trains two identically-initialized models — one via `train()` on a
pre-built `(X, y)` array, one via `train_streaming()` on the raw joint
vector — with the batch size forced larger than the dataset, so every epoch
is exactly one batch and row order (which the two shuffle differently:
`torch.randperm` vs. `np.random.default_rng().permutation`) can't affect a
mean-reduced loss or its gradient. Both the resulting loss curves and the
resulting models' predictions are asserted to match to `rtol=1e-4`. This is
what actually justifies the claim that the streaming path is a correctness-
preserving optimization rather than a different (if plausible-looking)
computation — the earlier synthetic/real end-to-end runs prove it doesn't
crash and produces sane-looking output, not that it computes the same
thing.

Early stopping has one easy-to-get-wrong edge case, caught while writing
its test: `best_val_loss` starts at `+inf`, so the *first* epoch always
counts as "improved," regardless of `min_delta` — an absurdly large
`min_delta` (used to force "never improves again" for the test) therefore
makes training stop after `1 + early_stopping_patience` epochs, not
`early_stopping_patience`. The test's original expectation assumed the
latter and had to be corrected; the implementation itself was already
right.

## GPU: available on this development machine, not yet wired up

`nvidia-smi` on this machine reports an **NVIDIA GeForce GTX 1650** (4096MiB
VRAM, driver supporting up to CUDA 13.2) — genuine GPU hardware is present.
The currently installed PyTorch build is CPU-only (`torch==2.13.0+cpu`,
confirmed by `torch.cuda.is_available() == False`), not because there's no
GPU, but because the CPU-only wheel got installed rather than a CUDA build.

A GTX 1650 is an entry-level/laptop-class card (not a large speedup like a
data-center GPU), and GRUs specifically don't parallelize across a GPU's
cores as well as CNNs/Transformers do — each of the 50 timesteps in a
sequence is still computed one after another either way, so the GPU's win
comes from processing more sequences per batch at once, not from speeding
up one sequence. A realistic estimate (**not measured** — this machine's
PyTorch is still CPU-only) is somewhere in the 3-8x range once batch size is
also raised (see the "Scaling plan" section above, which uses this same
range in a worked estimate) — plausibly better than that, but it should be
benchmarked once a CUDA build is installed rather than assumed. Its 4GB VRAM
is not a constraint either way — a batch of even 1,024 windows at this
model's shapes is a few MB, nowhere near the limit.

To use it: reinstall PyTorch with a CUDA-enabled build matching this
machine's driver (currently supports up to CUDA 13.2), in place of the
CPU-only wheel `requirements.txt`'s unpinned `torch` currently resolves to.
Whether this is worth doing on a machine a teammate might not have a GPU on
at all is a call for whoever's running the real-data training — the
CPU-only path still works, just slower, and `requirements.txt` staying
GPU-agnostic (unpinned `torch`) is what keeps the environment usable for
someone without one (see PLAN.md's environment-flexibility decision).
