"""Stricter test: run the detector, the attack-type labeller and Branch 2 on PRACTICE
attacks (made from normal traffic by scripts/build_injected_attacks.py) and, beside
them, on the real attack files.

Branch 1 (windowed forecast-residual detector) is fitted like the other checks
(train_1 validation + train_2, normal data only); Branch 2 (Isolation Forest +
plausibility limits over message rates) is loaded from --branch2-model. Nothing is
tuned here. Needs the cache from scripts/cache_gru_outputs.py including the injected
files (use --extra there) and the CSVs from build_injected_attacks.py.

Reading the tables: `drift_additive` is built so that the smooth-movement rule has NO
reason to recognise it (the ramp rides on the true wiggly signal), so it is expected to
be detected but labelled "replay". `drift_replace` is built like the real drift traces
looked, so it is a sanity check rather than independent evidence. Flooding and fuzzing
practice attacks are the independent test of Branch 2 (SynCAN has no real fuzzing file).

Usage:
    python scripts/injected_check.py <cache_dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from canids.classify import FALSE_ALARM, type_accuracy, type_confusion
from canids.config import GRID_STEP_SECONDS, RAW_DATA_DIR, SMOOTH_LOOKAHEAD_TICKS, SYNTHETIC_DATA_DIR
from canids.data.loader import load_attack
from canids.detector import NormalStream, count_alerts, detect, fit_detector
from canids.evaluate import event_metrics
from canids.fusion import fuse
from canids.models.isolation_forest import Branch2Model, detect_branch2

# file name in the cache -> the attack type it really is
FILES = {
    "injected_replay": "replay",
    "injected_drift_replace": "drift",
    "injected_drift_additive": "drift",
    "injected_plateau": "plateau",
    "injected_flooding": "flooding",
    "injected_fuzzing": "fuzzing",
    "injected_flooding_mild": "flooding",
    "injected_flooding_burst": "flooding",
    "injected_suppress_partial": "suppression",
    "replay": "replay",
    "drift": "drift",
    "plateau": "plateau",
    "flooding": "flooding",
}
REAL = {"replay", "drift", "plateau", "flooding"}
# built to slip under Branch 2's plausibility limits; only used for the "what the forest adds" table, not for type labels
EVASION = {"injected_flooding_mild", "injected_flooding_burst", "injected_suppress_partial"}
# the forest "earns its place" if on at least one evasion file it finds this many more events (at most MAX_EXTRA_FA more false alarms)
MIN_EXTRA_EVENTS, MAX_EXTRA_FA = 10, 1
LABELS = ["suppression", "plateau", "drift", "replay", "flooding", "fuzzing"]


def _load(cache: Path, name: str) -> dict[str, np.ndarray]:
    d = np.load(cache / f"{name}.npz")
    return {k: d[k] for k in d.files}


def _stream(d: dict[str, np.ndarray]) -> NormalStream:
    return NormalStream(d["residuals"].astype(float), d["staleness"].astype(float), d["values"].astype(float))


def _csv(name: str) -> Path:
    return SYNTHETIC_DATA_DIR / f"{name}.csv" if name.startswith("injected_") else RAW_DATA_DIR / f"syncan_test_{name}.csv"


def _tag(name: str) -> str:
    return "" if name in REAL else "  [practice]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("--branch2-model", type=Path, default=Path("models") / "isolation_forest_syncan_train1.joblib")
    parser.add_argument("--skip-branch2", action="store_true")
    args = parser.parse_args()

    det = fit_detector([_stream(_load(args.cache_dir, n)) for n in ("train1_val", "train2")])
    print(f"Branch 1 detector: window={det.window} tau={det.tau:.3f}; lookahead tested: 0 and {SMOOTH_LOOKAHEAD_TICKS} ticks\n")

    print("BRANCH 1 DETECTION (event level)")
    print(f"  {'file':26s} {'events':>6s} {'found':>6s} {'recall':>7s} {'alerts':>6s} {'false':>6s} {'FA/min':>7s} {'strictP':>8s}")
    outs = {}
    for name, true_type in FILES.items():
        if not (args.cache_dir / f"{name}.npz").exists():
            print(f"  {name:26s} (not cached -- skipped)")
            continue
        d = _load(args.cache_dir, name)
        out = detect(det, d["residuals"].astype(float), d["staleness"].astype(float), d["values"].astype(float))
        outs[name] = (out, d["gt"], true_type)
        m = event_metrics(out.flags, d["gt"], name, GRID_STEP_SECONDS)
        print(
            f"  {name:26s} {m.n_events:6d} {m.n_detected:6d} {m.recall:7.3f} {m.n_alerts:6d} {m.n_false_alarms:6d} "
            f"{m.false_alarms_per_minute:7.3f} {m.strict_precision:8.3f}{_tag(name)}"
        )

    for lookahead in (0, SMOOTH_LOOKAHEAD_TICKS):
        print(f"\nBRANCH 1 TYPE LABELS, lookahead {lookahead} ticks   (labelled correctly / total; what the wrong ones were called)")
        for name, (out, gt, true_type) in outs.items():
            if true_type in ("flooding", "fuzzing") or name in EVASION:
                continue  # Branch 1 has no label for these
            counts = type_confusion({true_type: out}, {true_type: gt}, det, lookahead_ticks=lookahead)
            ok, total = type_accuracy(counts, [true_type])[true_type]
            wrong = {pred: c for (true, pred), c in counts.items() if true == true_type and pred != true_type}
            print(f"  {name:26s} {ok:4d}/{total:<4d} = {ok / total:.3f}   wrong -> {wrong}{_tag(name)}")

    if args.skip_branch2 or not args.branch2_model.exists():
        print("\n(Branch 2 skipped: no model at", args.branch2_model, ")")
        return

    b2 = Branch2Model.load(args.branch2_model)
    print(f"\nBRANCH 2 (Isolation Forest + plausibility limits), model {args.branch2_model}")
    print("  event level, Branch 2 alone, split by component (forest only / limits only / both):")
    print(f"  {'file':26s} {'events':>6s} {'forest':>7s} {'limits':>7s} {'both':>6s} {'alerts':>6s} {'FA/min':>7s} {'strictP':>8s}")
    fused_outs = {}
    forest_rows = []
    for name, (out, gt, true_type) in outs.items():
        df = load_attack(_csv(name))
        full = detect_branch2(b2, df)
        ticks = np.arange(len(full.flags) - len(gt), len(full.flags))  # Branch 1 evaluates the last len(gt) ticks
        comp = full.at(ticks)
        limits = comp.rate_high | comp.oor_high
        m_f = event_metrics(comp.forest_flags, gt, name, GRID_STEP_SECONDS)
        m_l = event_metrics(limits, gt, name, GRID_STEP_SECONDS)
        m_b = event_metrics(comp.flags, gt, name, GRID_STEP_SECONDS)
        print(
            f"  {name:26s} {m_b.n_events:6d} {m_f.n_detected:7d} {m_l.n_detected:7d} {m_b.n_detected:6d} {m_b.n_alerts:6d} "
            f"{m_b.false_alarms_per_minute:7.3f} {m_b.strict_precision:8.3f}{_tag(name)}"
        )
        if name not in EVASION:
            fused_outs[name] = (fuse(out, full, ticks), gt, true_type)
        base = out.flags | limits
        m0, m1 = event_metrics(base, gt, name, GRID_STEP_SECONDS), event_metrics(base | comp.forest_flags, gt, name, GRID_STEP_SECONDS)
        forest_rows.append((name, m0, m1))

    print("\nWHAT THE FOREST ADDS: Branch 1 + plausibility limits, without vs with the Isolation Forest")
    print(f"  {'file':26s} {'events':>6s} | {'found':>6s} {'alerts':>6s} {'false':>6s} | {'found':>6s} {'alerts':>6s} {'false':>6s} | found ONLY thanks to the forest / extra false alarms")
    verdict = []
    for name, m0, m1 in forest_rows:
        extra, extra_fa = m1.n_detected - m0.n_detected, m1.n_false_alarms - m0.n_false_alarms
        print(
            f"  {name:26s} {m0.n_events:6d} | {m0.n_detected:6d} {m0.n_alerts:6d} {m0.n_false_alarms:6d} | "
            f"{m1.n_detected:6d} {m1.n_alerts:6d} {m1.n_false_alarms:6d} | {extra:+d} / {extra_fa:+d}{_tag(name)}{'  [evasion]' if name in EVASION else ''}"
        )
        if name in EVASION:
            verdict.append((name, extra, extra_fa))
    earned = [v for v in verdict if v[1] >= MIN_EXTRA_EVENTS and v[2] <= MAX_EXTRA_FA]
    print(
        f"  decision rule (fixed before running): the forest earns its place if on at least one [evasion] file it finds >= {MIN_EXTRA_EVENTS} more "
        f"events with <= {MAX_EXTRA_FA} extra false alarm -> {'YES: ' + ', '.join(v[0] for v in earned) if earned else 'NO'}"
    )

    # held-out NORMAL data: how often does each branch cry wolf?
    if (RAW_DATA_DIR / "syncan_test_normal.csv").exists() and (args.cache_dir / "normal.npz").exists():
        d = _load(args.cache_dir, "normal")
        out = detect(det, d["residuals"].astype(float), d["staleness"].astype(float), d["values"].astype(float))
        full = detect_branch2(b2, load_attack(RAW_DATA_DIR / "syncan_test_normal.csv"))
        minutes = len(full.flags) * GRID_STEP_SECONDS / 60
        print(
            f"\n  held-out NORMAL file ({minutes:.0f} min): Branch 1 {count_alerts(out.flags, 100)} alerts "
            f"({count_alerts(out.flags, 100) / minutes:.2f}/min), Branch 2 {count_alerts(full.flags, 100)} alerts "
            f"({count_alerts(full.flags, 100) / minutes:.2f}/min) [flags within 1 s merged]"
        )

    print("\nFUSED (Branch 1 OR Branch 2): labelled correctly / total, and the full label counts (alerts overlapping an attack)")
    for name, (fused, gt, true_type) in fused_outs.items():
        counts = type_confusion({true_type: fused}, {true_type: gt}, det)
        ok, total = type_accuracy(counts, [true_type])[true_type]
        called = {lab: counts.get((true_type, lab), 0) for lab in LABELS if counts.get((true_type, lab), 0)}
        fa = {lab: counts.get((FALSE_ALARM, lab), 0) for lab in LABELS if counts.get((FALSE_ALARM, lab), 0)}
        m = event_metrics(fused.flags, gt, name, GRID_STEP_SECONDS)
        print(
            f"  {name:26s} found {m.n_detected:3d}/{m.n_events:<3d} strictP {m.strict_precision:.3f}  labelled {ok:4d}/{total:<4d} = {ok / total:.3f}  "
            f"called {called}  false alarms {fa}{_tag(name)}"
        )


if __name__ == "__main__":
    main()
