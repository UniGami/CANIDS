"""Honest out-of-sample check of the stage-1 detector's settings.

The detector's settings (window length, false-alarm budget) were first chosen
while looking at whole attack files, so the headline numbers are optimistic.
This script removes that: every attack file is cut in half by time (at a
quiet spot, never through an attack). Settings are CHOSEN using only the
first halves, then the chosen settings are REPORTED on the second halves,
which played no part in the choice. The threshold itself is always set from
normal data only.

Needs the cache from scripts/cache_gru_outputs.py.

Usage:
    python scripts/detector_split_check.py <cache_dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from canids.config import GRID_STEP_SECONDS
from canids.relational import RelationalModel
from canids.detector import NormalStream, detect, fit_detector
from canids.evaluate import event_metrics, quiet_split_point

ATTACKS = ["plateau", "drift", "replay", "suppression", "flooding"]
WINDOWS = [25, 50, 100]
TARGETS = [0.05, 0.1, 0.3]
CURRENT = (50, 0.1)  # the settings chosen BEFORE this check, for comparison


def _load(cache: Path, name: str) -> dict[str, np.ndarray]:
    d = np.load(cache / f"{name}.npz")
    return {k: d[k] for k in d.files}


def _stream(d: dict[str, np.ndarray]) -> NormalStream:
    return NormalStream(d["residuals"].astype(float), d["staleness"].astype(float), d["values"].astype(float))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("--relational-model", type=Path, default=Path("models") / "relational_syncan_train1.json")
    parser.add_argument("--no-partner", action="store_true", help="leave the partner-consistency channel out of the Branch 1 detector")
    args = parser.parse_args()
    relational = None if args.no_partner or not args.relational_model.exists() else RelationalModel.load(args.relational_model)
    print("partner channel:", f"{len(relational.signals)} signals" if relational else "off")

    calib_streams = [_stream(_load(args.cache_dir, n)) for n in ("train1_val", "train2")]
    attacks = {a: _load(args.cache_dir, a) for a in ATTACKS}
    splits = {a: quiet_split_point(attacks[a]["gt"]) for a in ATTACKS}
    print("split points (tick of first half end / file length):", {a: f"{splits[a]}/{len(attacks[a]['gt'])}" for a in ATTACKS})

    # results[(window, target)][attack] = (metrics on first half, metrics on second half)
    results: dict[tuple[int, float], dict[str, tuple]] = {}
    for window in WINDOWS:
        for target in TARGETS:
            det = fit_detector(calib_streams, window=window, target_alerts_per_minute=target, relational=relational)
            per_attack = {}
            for a in ATTACKS:
                d = attacks[a]
                out = detect(det, d["residuals"].astype(float), d["staleness"].astype(float), d["values"].astype(float))
                t = splits[a]
                first = event_metrics(out.flags[:t], d["gt"][:t], a, GRID_STEP_SECONDS)
                second = event_metrics(out.flags[t:], d["gt"][t:], a, GRID_STEP_SECONDS)
                per_attack[a] = (first, second)
            results[(window, target)] = per_attack

    def mean_f1(cfg: tuple[int, float], half: int) -> float:
        vals = [results[cfg][a][half].strict_f1 for a in ATTACKS]
        return float(np.nanmean(vals)) if not all(np.isnan(vals)) else float("nan")

    print("\nStep 1 -- choose settings using ONLY the first halves (average strict F1 over the 5 attacks):")
    print(f"  {'window':>6s} {'alerts/min':>10s} {'first-half F1':>14s}")
    for cfg in results:
        print(f"  {cfg[0]:6d} {cfg[1]:10.2f} {mean_f1(cfg, 0):14.3f}")
    chosen = max(results, key=lambda c: mean_f1(c, 0))
    print(f"\n  -> chosen from first halves: window={chosen[0]}, alerts/min target={chosen[1]}")

    def show(cfg: tuple[int, float], label: str) -> None:
        print(f"\nStep 2 -- {label}: window={cfg[0]}, target={cfg[1]}  (SECOND halves only, never used for choosing)")
        print(f"  {'attack':12s} {'events':>6s} {'found':>6s} {'recall':>7s} {'alerts':>6s} {'false':>6s} {'FA/min':>7s} {'strictP':>8s} {'strictF1':>9s}")
        for a in ATTACKS:
            m = results[cfg][a][1]
            print(
                f"  {a:12s} {m.n_events:6d} {m.n_detected:6d} {m.recall:7.3f} {m.n_alerts:6d} {m.n_false_alarms:6d} "
                f"{m.false_alarms_per_minute:7.3f} {m.strict_precision:8.3f} {m.strict_f1:9.3f}"
            )
        print(f"  average strict F1 on second halves: {mean_f1(cfg, 1):.3f}   (first halves: {mean_f1(cfg, 0):.3f})")

    show(chosen, "settings chosen on first halves")
    if chosen != CURRENT:
        show(CURRENT, "the settings used before this check")


if __name__ == "__main__":
    main()
