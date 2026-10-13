"""Out-of-sample check of the alert start/stop refinement (`runs.refine_alert_bounds`).

Most tick-level errors sit at the edges of attacks: an alert starts a few ticks after the attack began (the running
mean needs time to build up) and runs on for about one window after it ended. The refinement walks each alert's
start back to where its score first rose above `alpha * tau` and cuts `trim` ticks off its end. It creates and removes
no alerts, so the number of attacks found does not change; only each alert's extent does.

Like scripts/detector_split_check.py: every attack file is cut in half at a quiet spot; the settings (alpha, how far
back, how much tail) are CHOSEN on the first halves only (best average tick F1 over the 5 attacks, as long as the
average event-level strict F1 does not drop by more than 0.01), and REPORTED on the second halves. The detector's
threshold itself comes from normal data only. Judged on the fused (Branch 1 OR Branch 2) verdict.

Usage:
    python scripts/boundary_split_check.py <cache_dir> --calibration-files train1_val train2L train3L
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np

from canids.config import GRID_STEP_SECONDS, RAW_DATA_DIR
from canids.data.loader import load_attack
from canids.detector import NormalStream, detect, fit_detector
from canids.evaluate import detection_metrics, event_metrics, quiet_split_point
from canids.fusion import fuse
from canids.models.rate_limits import Branch2Model, detect_branch2
from canids.relational import RelationalModel
from canids.runs import refine_alert_bounds

ATTACKS = ["plateau", "drift", "replay", "suppression", "flooding"]
ALPHAS = [0.5, 0.75, 0.9]
BACK = [100, 300]
TRIMS = [0, 25, 50, 75, 100]
NONE = (1.0, 0, 0)  # alpha, max_back, trim: leaves every alert as it was


def _load(cache: Path, name: str) -> dict[str, np.ndarray]:
    d = np.load(cache / f"{name}.npz")
    return {k: d[k] for k in d.files}


def _stream(d: dict[str, np.ndarray]) -> NormalStream:
    return NormalStream(d["residuals"].astype(float), d["staleness"].astype(float), d["values"].astype(float))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("--calibration-files", nargs="*", default=["train1_val", "train2"])
    parser.add_argument("--relational-model", type=Path, default=Path("models") / "relational_syncan_train1.json")
    parser.add_argument("--branch2-model", type=Path, default=Path("models") / "branch2_limits_syncan_train1.json")
    args = parser.parse_args()

    relational = RelationalModel.load(args.relational_model)
    det = fit_detector([_stream(_load(args.cache_dir, n)) for n in args.calibration_files], relational=relational)
    b2 = Branch2Model.load(args.branch2_model)

    data = {}
    for a in ATTACKS:
        d = _load(args.cache_dir, a)
        gt = d["gt"].astype(bool)
        out = detect(det, d["residuals"].astype(float), d["staleness"].astype(float), d["values"].astype(float))
        full = detect_branch2(b2, load_attack(RAW_DATA_DIR / f"syncan_test_{a}.csv"))
        ticks = np.arange(len(full.flags) - len(gt), len(full.flags))
        data[a] = (fuse(out, full, ticks), gt, quiet_split_point(gt))

    configs = [NONE] + [(al, L, tr) for al, L, tr in product(ALPHAS, BACK, TRIMS) if (L, tr) != (0, 0)]

    def evaluate(cfg):
        """per attack: (tick metrics first half, tick metrics second half, event metrics first, event metrics second)"""
        res = {}
        for a, (fused, gt, t) in data.items():
            alpha, back, trim = cfg
            flags = refine_alert_bounds(fused.flags, fused.detector.score, det.tau, alpha, back, trim, 10)
            res[a] = (
                detection_metrics(flags[:t], gt[:t], a, 0.0), detection_metrics(flags[t:], gt[t:], a, 0.0),
                event_metrics(flags[:t], gt[:t], a, GRID_STEP_SECONDS), event_metrics(flags[t:], gt[t:], a, GRID_STEP_SECONDS),
            )
        return res

    results = {cfg: evaluate(cfg) for cfg in configs}

    def mean(cfg, half, kind):
        i = {("tick", 0): 0, ("tick", 1): 1, ("event", 0): 2, ("event", 1): 3}[(kind, half)]
        attr = "f1" if kind == "tick" else "strict_f1"
        return float(np.nanmean([getattr(results[cfg][a][i], attr) for a in ATTACKS]))

    base_event = mean(NONE, 0, "event")
    print(f"detector tau={det.tau:.3f}; settings = (alpha, max ticks back, ticks trimmed off the end)")
    print("\nStep 1 -- choose using ONLY first halves: average tick F1 (event strict F1 must stay within 0.01 of unrefined)")
    print(f"  {'settings':>20s} {'tick F1':>8s} {'event F1':>9s}")
    ok = []
    for cfg in configs:
        tf, ef = mean(cfg, 0, "tick"), mean(cfg, 0, "event")
        allowed = ef >= base_event - 0.01
        if allowed:
            ok.append(cfg)
        print(f"  {str(cfg):>20s} {tf:8.3f} {ef:9.3f}{'' if allowed else '   (event F1 dropped too much)'}")
    chosen = max(ok, key=lambda c: mean(c, 0, "tick"))
    print(f"\n  -> chosen from first halves: {chosen}")

    def show(cfg, label):
        print(f"\nStep 2 -- {label}: {cfg}  (SECOND halves only, never used for choosing)")
        print(f"  {'attack':12s} {'tick P':>7s} {'tick R':>7s} {'tick F1':>8s} | {'events':>6s} {'found':>6s} {'alerts':>6s} {'false':>6s} {'strictF1':>9s}")
        for a in ATTACKS:
            tm, em = results[cfg][a][1], results[cfg][a][3]
            print(
                f"  {a:12s} {tm.precision:7.3f} {tm.recall:7.3f} {tm.f1:8.3f} | {em.n_events:6d} {em.n_detected:6d} "
                f"{em.n_alerts:6d} {em.n_false_alarms:6d} {em.strict_f1:9.3f}"
            )
        ps = np.nanmean([results[cfg][a][1].precision for a in ATTACKS])
        rs = np.nanmean([results[cfg][a][1].recall for a in ATTACKS])
        print(f"  average: tick precision {ps:.3f}  recall {rs:.3f}  F1 {mean(cfg, 1, 'tick'):.3f}  event strict F1 {mean(cfg, 1, 'event'):.3f}")

    show(NONE, "no refinement (as now)")
    show(chosen, "refinement with the chosen settings")


if __name__ == "__main__":
    main()
