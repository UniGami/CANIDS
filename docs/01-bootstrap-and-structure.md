# Steps 1–2: Repo, Environment, and Project Structure

## Goal

Get from "just `claude.md`, no code" to a project that's importable, testable,
and buildable by two people with different local tooling, before writing any
pipeline logic.

## Technical decisions

**Data strategy — synthetic first, real SynCAN later, one seam.**
claude.md left open whether to scaffold against real SynCAN CSVs (not yet
uploaded) or synthetic placeholder data. Chose synthetic: it unblocks
implementation immediately and, because `data/synthetic.py` produces exactly
the schema `data/loader.py` expects, swapping to real SynCAN later is a
one-line change (point `loader.py` at `data/raw/`) — no pipeline code changes.

**Framework — PyTorch, for both GRU and TCN.** claude.md named the models
(GRU, TCN, Isolation Forest) but not a deep learning framework. PyTorch was
chosen over TensorFlow/Keras for more direct control over a custom TCN's
dilated causal convolutions, which is fiddlier to hand-roll in Keras's
higher-level API.

**Environment — `requirements.txt` as the single source of truth, conda as a
thin wrapper.** The user works in Anaconda; a teammate might not. Making the
project *require* conda would block the teammate; making it conda-agnostic
loses the convenience the user already has. The resolution:
- `requirements.txt` — plain pip-installable dependency list. This is what
  everything else defers to.
- `environment.yml` — a minimal conda env (`python=3.11 + pip`) whose only
  job is to `pip install -r requirements.txt` and `pip install -e .`. Conda
  is just a Python-version-and-venv manager here, not a dependency manager.
- Either person runs `pip install -r requirements.txt` in their own venv, or
  `conda env create -f environment.yml` — same dependency versions either
  way, because both paths read the same file.

**Editable install via `pyproject.toml` + `src/` layout.** `canids` is
installed with `pip install -e .` so scripts and tests import it as a real
package (`from canids.registry import ...`) instead of relying on relative
path hacks or `sys.path` manipulation. The `src/` layout (code lives in
`src/canids/`, not a bare `canids/` next to `tests/`) prevents an import-path
foot-gun where `import canids` could accidentally resolve to the working
directory instead of the installed package.

**Git ignoring generated data.** `.gitignore` excludes `data/raw/` and
`data/processed/` entirely (real SynCAN CSVs and cached tensors — large,
regenerable-or-external, shouldn't live in git). `data/synthetic/` is
special-cased: `data/synthetic/*` is ignored but `!data/synthetic/.gitkeep`
keeps the empty directory tracked, since the synthetic CSVs it holds are
generated on demand by `scripts/generate_synthetic_data.py` and don't need
to be committed, but the directory itself needs to exist for that script to
write into without a first-run `mkdir`.

## Resulting structure

```
CANIDS/
  claude.md              # architecture spec (source of truth for the design)
  PLAN.md                 # this implementation roadmap, with a progress checklist
  requirements.txt / environment.yml / pyproject.toml
  src/canids/             # the actual package (see later docs for each module)
  scripts/                 # thin CLI entry points, e.g. generate_synthetic_data.py
  tests/                    # pytest, one file per module
  data/raw/ synthetic/ processed/
```

No functions to document here — this step is purely repo/tooling scaffolding.
See [02-signal-registry.md](02-signal-registry.md) onward for the first real code.
