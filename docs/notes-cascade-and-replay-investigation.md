## Cross-cutting note: cross-signal cascade false positives, a rule-mislabeling bug, and why replay detection was effectively switched off

**Status:** investigated, fixed, and verified against real data. This
session traced *why* Branch 1's real-data false positives are so high, and
specifically dug into why the `replay` rule (the hardest attack type, and
the main thing we actually care about detecting) almost never fires. We
found four distinct, evidenced problems, one of which (replay) turned out
to be the headline result: **replay detection wasn't just weak, it was
structurally impossible to trigger under the current confidence gate.**
All three fixable findings (replay's dead gate, plateau's mislabeling, and
cross-signal cascade) have since been implemented in
`src/canids/attribution/rules.py` / `src/canids/models/naive.py` /
`src/canids/config.py`, covered by 13 new tests plus 3 corrected existing
ones (all 146 repo tests pass), and verified end-to-end against the real
trained checkpoint — see "Implementation and results" below for the actual
before/after numbers, including one honest nuance the fix surfaced on the
suppression file that wasn't part of the original three findings.

### Key terms, in plain English

If you already know this codebase, skip ahead — this section exists so the
findings below don't require re-deriving the vocabulary.

- **GRU (Gated Recurrent Unit):** the neural network used to forecast what
  the CAN bus "should" look like one tick into the future, based on the
  recent past. It reads a short window of recent signal values and outputs
  one prediction per signal for the next tick.
- **Joint state vector / shared hidden state:** instead of one small model
  per signal, this project uses **one single GRU** that reads *all 20
  signals at once* and predicts all 20 of them at once. Internally the GRU
  keeps one running summary of "what's been happening" (the *hidden
  state*) that every signal's prediction is computed from. This detail
  matters a lot below.
- **Residual:** `actual value − predicted value`. A big residual means the
  model was surprised — either something unusual is happening, or the
  model's forecast for that signal just isn't very good.
- **Confidence gate:** before trusting a signal's residuals for detection,
  the code checks whether the GRU actually forecasts that signal better
  than a trivial "predict no change" baseline, on clean (non-attack)
  validation data. If it doesn't beat that baseline by a comfortable
  margin, the signal is marked `LOW CONF` and its residuals are **ignored**
  by three of the four detection rules (see below). Only 4 of this
  project's 20 signals currently pass this gate.
- **Correlation graph:** built once, offline, from normal driving data
  only. It records which *pairs* of signals reliably move together (e.g.
  wheel-speed-like signals that rise and fall in lockstep). This is used by
  the `replay` rule as a sanity check: "did this signal *and* its usual
  partner both go anomalous at the same time?"
- **Calibration / thresholds:** before evaluating any attack data, the
  pipeline looks at *only* clean validation data to decide "how big a
  residual is normal" per signal. Anything past that learned threshold
  counts as suspicious.
- **CUSUM (cumulative sum):** the statistic behind the `drift` rule. Instead
  of asking "is this one tick's residual too big," it keeps a running total
  of residuals over time, so it can catch a *slow, sustained* drift that no
  single tick would flag on its own.
- **The four attribution rules**, checked in this priority order — the
  first one that matches for a given signal at a given moment "wins" and
  becomes that signal's label:
  1. **suppression** — a signal stops transmitting for longer than its
     normal expected gap (based on the "staleness" counter — ticks since
     last real transmission). The only rule that doesn't use the GRU at
     all, so it's untouched by anything about model quality.
  2. **plateau** — a signal's real value is frozen (bit-for-bit identical
     across repeated transmissions) while the model's prediction keeps
     drifting away from it.
  3. **drift** — a signal's residual keeps trending in the same direction
     for a sustained period (the CUSUM statistic above).
  4. **replay** — a signal *and one of its correlation-graph partners*
     both go anomalous at the same time, with no plateau/drift signature —
     because a "replayed" (rebroadcast) window of the bus tends to make a
     whole cluster of related signals look anomalous together, since
     they're all replaying real, self-consistent past data.
- **Ground truth / ground-truth window:** the actual time range during
  which the test file really was under attack, taken from the file's
  `Label` column. Important real-SynCAN quirk: the `Label` column marks
  *every* signal's rows as "attack" during that window, not just the one
  signal actually being attacked — so ground truth tells you *when* an
  attack happened, but never *which signal* was the real target. Figuring
  that out took separate small scripts, described per finding below.
- **Precision / recall:** precision = "of everything we flagged, how much
  was a real attack" (low precision = too many false alarms). Recall = "of
  all the real attack moments, how many did we catch" (low recall = missed
  attacks). A useless detector can get either one to 100% by flagging
  everything or nothing — both numbers have to be read together.
- **True positive / false positive / false negative (TP / FP / FN):** TP =
  correctly flagged an attack tick. FP = flagged a normal tick as an
  attack. FN = missed a real attack tick.

### How this was investigated

Everything below was produced with `scripts/inspect_pipeline.py --train`
(already extended this session with a "Detection Summary" that runs the
full detection pipeline against a whole real SynCAN attack file and prints,
per signal, exactly what it was labeled and how often) plus a handful of
small, throwaway analysis scripts written for this investigation only (not
committed — see "Files used" at the end). All runs reuse the already-trained
checkpoint `models/gru_syncan_train1.pt` (no retraining), so every number
below is reproducible by re-running the same commands.

---

### Finding 1: A real attack on one signal makes *other, unrelated* signals look attacked too ("cascade misattribution")

**The file:** `data/raw/syncan_test_suppression.csv` (a real *suppression*
attack — one signal's messages are cut off the bus entirely for a while).

We first independently confirmed the attack was real by checking, outside
the model entirely, how often each of the 10 CAN IDs actually transmitted
before/during/after the labeled window:

| ID | messages/sec before | messages/sec **during** | messages/sec after |
|---|---|---|---|
| id8 | 66.58 | **0.00** | 66.67 |
| all other 9 IDs | unchanged | unchanged | unchanged |

`id8` goes completely silent for exactly the labeled attack window
(`t=[76537.48, 80935.37]`) while everything else keeps transmitting
normally — a real, unambiguous suppression attack on `id8`.

Running the full pipeline on this file:

| | count |
|---|---|
| ticks in the file | 449,994 |
| ticks genuinely under attack (ground truth) | 79,714 |
| ticks the detector flagged | 447,332 (99.4% of the whole file) |
| correctly flagged attack ticks | 79,713 |
| **false alarms** | **367,619** |
| missed attacks | 1 |
| `suppression` rule fired | 170,502 times |
| `drift` rule fired | **1,377,477 times** |

`id8` itself gets the correct label, `suppression`, every time — the
staleness-based rule doesn't depend on the GRU at all, so it's immune to
whatever else is going on. But look at what else fired `drift`:

| signal | confidence | % of the whole file falsely labeled `drift` |
|---|---|---|
| id3_sig1 | passes the gate ("trustworthy") | **96%** |
| id7_sig2 | passes the gate | **82%** |
| id1_sig2 | passes the gate | **77%** |
| id6_sig2 | passes the gate | **48%** |

None of these four signals are `id8`. None of them are suppressed, frozen,
or drifting for real — they're forecast perfectly normally on clean data.
But for nearly the entire file, they get falsely accused of drifting.

**Why:** because there's one shared GRU hidden state reading *all* 20
signals at once, `id8` going silent doesn't just corrupt `id8`'s own
forecast — it corrupts the one shared internal summary the GRU uses to
predict *every other signal too*. The model was never trained on data where
one signal freezes while everything else keeps moving (that never happens
in normal driving), so once it does happen, the model's forecasts for
unrelated signals go subtly but persistently wrong — exactly the kind of
sustained, large deviation the `drift` rule is designed to catch, just
aimed at the wrong signal.

We reproduced this a second time on a completely different attack (real
*drift* attack, `data/raw/syncan_test_drift.csv`) and found the same
pattern: 3 of the same 4 signals falsely fire `drift` across a huge
fraction of that file too, while the file's *actual* target signals
(identified below) get no correct detection at all.

**The real target of the drift file, and a much worse discovery.** Using a
statistical trend-detection check directly on the raw signal values (no
model involved — just "does this value hold steady, then shift a lot, then
slowly recover, only during the labeled window"), we found the genuine
targets: `id5_sig2`, `id10_sig3`, and `id6_sig1` (note: signal *1* on id6,
a different signal than the `id6_sig2` above). All three drop from a
near-constant baseline (~0.997) down to ~0.69 right when the attack starts,
then gradually climb back toward normal across the rest of the window — a
textbook drift signature, and the three signals move in near-identical
lockstep (they're strongly correlated with each other in normal driving
too, confirmed by the correlation graph).

**All three of those real targets are `LOW CONF`.** Since `drift` (like
`plateau` and `replay`) is switched off entirely for `LOW CONF` signals,
**this specific attack can never be correctly detected, no matter what.**
The file's measured ~50% recall isn't partial credit for catching part of
a real attack — it's 100% coincidence: unrelated cascade noise from
`id1_sig2` / `id3_sig1` / `id7_sig2` happening to overlap the same time
window as the real (undetectable) attack.

---

### Finding 2: `plateau` attacks get the right signal, but the wrong label

**The file:** `data/raw/syncan_test_plateau.csv`.

Same 4 "trustworthy" signals dominate the firing counts again — but this
time, a self-contained check (looking for unusually long streaks of
*bit-for-bit identical* consecutive real transmissions, which is exactly
what a frozen/plateaued value looks like, and doesn't depend on picking a
comparison window at all) shows these ARE the genuine attack targets this
time:

| signal | longest identical-value streak outside the attack window | longest streak **inside** the window |
|---|---|---|
| id1_sig2 | 1 | 554 |
| id7_sig2 | 2 | 551 |
| id3_sig1 | 2 | 535 |
| id6_sig2 | 1 | 251 |

So unlike Finding 1, this isn't a cascade onto innocent bystanders — the
model IS reacting to a real attack on the right signals. The problem is
which rule claims credit: these signals mostly get labeled `drift`
(58,896–124,100 times each) rather than `plateau` (1,101–7,414 times each),
even though "value frozen solid for 500+ transmissions" is a much cleaner
match for `plateau`'s own definition.

**Why:** a genuinely frozen signal satisfies *both* rules' conditions at
once (flat value + growing residual = plateau's definition; a sustained,
one-directional residual trend = drift's definition, and a frozen value
naturally produces exactly that trend). `drift`'s CUSUM statistic is a
running total, so once it crosses its threshold it tends to *stay* above
it for a long stretch, even through ticks where `plateau`'s stricter,
tick-by-tick "is it flat AND is the residual over threshold right now"
check happens to momentarily fail. Since `drift` only loses the priority
race on ticks where `plateau` is *also* actively firing, `drift` ends up
winning most of the individual ticks by default — not by design, just by
being the less brittle statistic. Net effect: real plateau attacks are
detected, but reported as the wrong attack type.

---

### Finding 3: `flooding` doesn't have one clean target signal — and that's expected

**The file:** `data/raw/syncan_test_flooding.csv`.

Checking each ID's transmission rate before vs. during the attack:

| ID | rate increase during attack |
|---|---|
| id3, id5 | ~1.30x |
| id9, id4 | ~1.28x |
| id6 | ~1.21x |
| everything else | ~1.09x – 1.15x |

No dramatic spike on any single ID — just a broad, roughly uniform bump
across all 10. That's expected: per this project's locked architecture,
`flooding`/DoS-style attacks are Branch 2's job (a planned Isolation
Forest over rate/volume features), not Branch 1's. There's no "correct"
single-signal label for Branch 1's rules to land on here at all — the
attack doesn't have one.

---

### Finding 4 (the headline result): replay detection is not weak — it's switched off

This is what we actually set out to answer this session: **how does the
current pipeline do at catching replay attacks, the hardest and most
important attack type?**

**The file:** `data/raw/syncan_test_replay.csv`. Baseline run (current
default settings):

| | count |
|---|---|
| ground-truth attack ticks | 59,202 |
| correctly flagged | 7,559 (recall **12.8%**) |
| false alarms | 10,068 |
| **`replay` rule fired** | **0 times** — not once, on its own dedicated file |

Every single detection on this file comes from the same cascade effect as
Finding 1 (drift firing on `id1_sig2`/`id3_sig1`/`id7_sig2`) — none of it
is `replay` actually working.

**We proved this is not just bad luck — it's mathematically guaranteed to
be zero.** `replay` requires a signal's own residual to be too big, *and*
at least one of its correlation-graph partners' residuals to also be too
big at the same tick — and both checks are blocked entirely for `LOW CONF`
signals. Checking the actual numbers:

- Only 4 signals ever pass the confidence gate: `id1_sig2`, `id3_sig1`,
  `id6_sig2`, `id7_sig2`.
- **Three of those four have zero correlation-graph partners at all.**
- The fourth (`id6_sig2`) has exactly one partner (`id10_sig2`) — but that
  partner is `LOW CONF`, so its residual is forced to "not anomalous" by
  the gate regardless of what actually happens.

So there is no possible combination of events on any file that could ever
make `replay` fire. This has nothing to do with attack severity or
calibration percentile — it's a fixed property of which signals happen to
pass the confidence gate and which pairs happen to be in the correlation
graph, for this specific trained model.

**Is this a fragile coincidence, or a deep mismatch?** We tested by
gradually relaxing the confidence gate's strictness:

| how strict the gate is | how many signals pass | is a `replay` firing even possible? |
|---|---|---|
| current setting | 4 | No |
| slightly relaxed | 5 (adds `id10_sig2`) | Yes — but only one specific pair |
| gate removed completely | all 20 | Yes — every pair |

A tiny relaxation is enough to flip it from "impossible" to "possible" —
so it's a coincidence, not a deep design flaw. But testing that one small
relaxation directly still produced **zero** `replay` firings, because the
one pair it unlocked (`id6_sig2` / `id10_sig2`) just isn't the pair
actually involved in this particular attack.

**Removing the gate entirely tells the real story.** With every signal
eligible, `replay` fired 118,938 times — and **95.9% of those firings
(114,013) landed inside the real attack window.** That's a strong, useful
signal, not noise. Overall detection on the file jumped from 12.8% recall
to **86.1%** recall (missed attacks dropped from 51,643 to 8,218), at the
cost of more false alarms (since removing the gate everywhere also
re-enables `drift`/`plateau` on 15 other signals that genuinely do have
unreliable forecasts).

Crucially, almost none of the newly-firing signals are the 4
"trustworthy" ones — the real replay signal comes from **12 of the
correlation graph's 13 pairs**, each showing 90–98% precision on its own:

| signal | replay firings | % inside the real attack window |
|---|---|---|
| id2_sig3 | 11,100 | 97.7% |
| id8_sig1 | 11,096 | 97.7% |
| id7_sig1 | 8,004 | 97.0% |
| id2_sig1 | 8,445 | 96.6% |
| id5_sig1 | 13,927 | 96.2% |
| id4_sig1 | 14,088 | 96.0% |
| id10_sig1 | 8,610 | 89.8% |
| *(7 more, all similarly high)* | | |

This makes sense once you know what a replay attack actually is: it
rebroadcasts a whole chunk of real, self-consistent past traffic. Because
it's genuine historical data (not random noise), *multiple correlated
signals go anomalous together*, all internally consistent with each
other — precisely the two-signal-agreement pattern `replay`'s rule is
built to catch. That agreement requirement is already a strong, effective
filter on its own. Making every one of those signals *also* clear a
confidence bar tuned for `drift`/`plateau`'s very different needs doesn't
add extra safety — it just excludes nearly the entire correlation graph,
because most correlated pairs happen to involve signals the GRU doesn't
forecast especially well individually (which is irrelevant to whether
*two* of them spiking together is meaningful).

**Bottom line for the stated goal:** the current pipeline detects zero
real replay attacks, by construction — not "rarely," not "only the
obvious ones," actually zero, always, on any file. The fix looks
low-risk and well-targeted: replay doesn't need the same confidence gate
as drift/plateau, because its own two-signal corroboration check is
already doing that job, evidently well (~96% precision when unblocked).

---

### How this connects to what was already known

Two things flagged as unexplained in `docs/notes-false-positive-investigation.md`
and `docs/notes-real-data-scaling.md` are now explained:

- *"`suppression` stayed essentially completely flat... across every
  single lever tried this session... a genuinely unresolved anomaly."*
  Now explained: those false positives were never suppression's own
  detection logic misfiring — they're `drift` firing on four completely
  unrelated signals via the cascade effect in Finding 1. Nothing that
  tunes suppression's own thresholds could ever have moved that number.
- *"`drift` fires heavily on OTHER attack types' ground-truth windows
  too"* (previously attributed only to generic "OR-across-20-signals
  fusion saturation"). Now explained more precisely: it's the same 3-4
  signals, every time, regardless of which file — because they're the
  only signals capable of firing `drift`/`plateau` at all, and whatever
  attack is actually happening elsewhere in the file corrupts their
  forecasts too.

Also worth noting: every aggregate number produced this session
(precision/recall per file) landed exactly on the existing
`persist=12, decay=0.0001` row already recorded in
`notes-false-positive-investigation.md`'s results table — confirming this
investigation's tooling reproduces prior results exactly, just with new
per-signal visibility that wasn't captured before.

### Implementation and results

All three fixes below stay entirely within the attribution/rules layer —
none of them touch the locked GRU architecture. Each was tested with new
unit tests (synthetic, no trained model needed) and then verified by
re-running `scripts/inspect_pipeline.py --train` against the real
checkpoint on the same three real SynCAN files used above.

**1. `replay` gets its own confidence rule, separate from `drift`/`plateau`.**
`src/canids/models/naive.py` gained `confidence_weight()`: a graduated
version of the existing `confidence_gate()` that returns a continuous
per-signal value in `[CONFIDENCE_WEIGHT_FLOOR, 1.0]` instead of a hard
True/False. `detect_replay()` (`attribution/rules.py`) now takes this
`confidence_weight` instead of the boolean `confidence_mask`, with a
partner's contribution to corroboration summed and weighted
(`REPLAY_MIN_SIGNAL_WEIGHT`/`REPLAY_MIN_PARTNER_STRENGTH` in `config.py`)
instead of required to individually pass the same hard gate drift/plateau
use. Real-data result on `syncan_test_replay.csv`:

| | before | after |
|---|---|---|
| `replay` firings | 0 | 148,164 |
| recall | 12.8% | **72.6%** |
| precision | 42.9% | **77.9%** |

Both precision and recall improved substantially — replay is now the
dominant, correct detection mechanism on its own attack file.

**2. `plateau`'s own condition is more robust, so it stops losing to `drift` by default.**
New `frozen_streak_length()` generalizes the old single-tick
`values[t]==values[t-1]` check into a real run length
(`PLATEAU_MIN_FROZEN_STREAK_TICKS=8`), and `detect_plateau`'s residual
check is now streak-scoped (fires if the residual exceeded threshold at
*any* point since the current frozen run began, not only at the current
tick) so a momentary residual dip mid-run no longer breaks detection.
Rule priority itself was already correct and needed no change. Real-data
result on `syncan_test_plateau.csv`:

| | before | after |
|---|---|---|
| recall | 65.0% | **92.4%** |
| precision | 22.7% | **29.0%** |
| `plateau` firings on the 4 target signals | 1,101–6,254 each | 1,261–7,759 each |

Both precision and recall improved; `plateau` now correctly claims a much
larger share of its own attack's ticks instead of losing them to `drift`.

**3. A cross-signal cascade discount.** New `cascade_strength()` computes,
per tick, the strongest "how far past its own threshold" ratio among
*other* signals independently firing `suppression` or `plateau` at the
same moment (deliberately never `drift`, to avoid suppressing a genuine
simultaneous multi-signal drift attack); `attribute()` uses this to
discount a signal's `drift` firing when another signal's independent
evidence is at least `CASCADE_DISCOUNT_STRENGTH_THRESHOLD=2.0`× past its
own threshold. Real-data result on `syncan_test_suppression.csv`:

| | before | after |
|---|---|---|
| overall flagged / TP / FP | 447,332 / 79,713 / 367,619 | **unchanged** |
| `drift` firings on the 4 cascade-affected signals | 214,600–431,740 each | 162,191–361,042 each (down 15–25%) |

**An honest nuance this surfaced, not one of the original three findings:**
the per-signal mislabeling improved (less wrong `drift`), but the overall
false-positive *count* on this file didn't move, because `replay` (now
unblocked by fix #1) picked up nearly the same tick coverage `drift` lost.
During a suppression event, many correlated signal *pairs* go anomalous
together too, which satisfies `replay`'s own two-signal corroboration
check even though no real replay attack is happening on this file — a
second-order cascade effect through `replay`'s own mechanism. Net effect:
attribution got more accurate (correct rule labels matter for the
reporting/attribution goal this project cares about), but this file's raw
tick-level precision/recall didn't improve — worth knowing rather than
overselling.

All 146 tests in the repo pass (`pytest tests/`), including 13 new tests
(`frozen_streak_length`, the streak-scoped `detect_plateau` fix and its
short-repeat guard-rail, `cascade_strength` and its two `attribute()`-level
regression/guard-rail tests, two `confidence_weight` tests in
`tests/test_naive.py`, and two `detect_replay` tests for the newly-unblocked
LOW CONF corroboration path) and 3 corrected existing ones (one flagged by
this investigation's own planning, `test_confidence_mask_suppresses_residual_rules_but_not_suppression`;
one caught only by hand-tracing during implementation,
`test_detect_plateau_requires_both_flat_value_and_residual_exceeds`; and
`test_attribute_priority_order_suppression_beats_everything`, whose short
fixture needed an explicit `plateau_min_frozen_streak_ticks` override).

### Files changed

- `src/canids/attribution/rules.py` — `frozen_streak_length`,
  `cascade_strength`, rewritten `detect_plateau`/`detect_replay`, and
  `attribute()`'s new `plateau_min_frozen_streak_ticks`/
  `cascade_discount_strength_threshold`/`confidence_weight` parameters.
- `src/canids/models/naive.py` — new `confidence_weight()`.
- `src/canids/config.py` — `CONFIDENCE_WEIGHT_FLOOR`,
  `PLATEAU_MIN_FROZEN_STREAK_TICKS`, `CASCADE_DISCOUNT_STRENGTH_THRESHOLD`,
  `REPLAY_MIN_SIGNAL_WEIGHT`, `REPLAY_MIN_PARTNER_STRENGTH` — all flagged
  as starting values needing a real-data sweep before being trusted as
  final, same caveat every other empirically-derived constant in that file
  carries.
- `src/canids/evaluate.py`, `scripts/run_detector.py`,
  `scripts/run_evaluation.py`, `scripts/inspect_pipeline.py` — threaded
  `confidence_weight` through to `attribute()` alongside the existing
  `confidence_mask`.
- `tests/test_attribution.py`, `tests/test_naive.py` — new and corrected
  tests described above.

### Files used for investigation (not committed — throwaway, one-off analysis)

- `check_attribution.py`, `check_drift_target.py` / `check_drift_target2.py`,
  `check_plateau_flooding_targets.py` / `_targets2.py`,
  `check_replay_deadend.py`, `check_replay_relaxed.py`,
  `check_replay_surgical.py` — session scratchpad only.
- `scripts/inspect_pipeline.py`'s `--train` Detection Summary — the
  reusable tool that produced every aggregate number in this document.
