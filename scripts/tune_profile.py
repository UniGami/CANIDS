"""Choose a forecaster's detector settings (its pipeline profile) on the FIRST halves of the attack files and report the
SECOND halves, which played no part in the choice.

Why per forecaster: the detector's settings were first chosen around the GRU's errors. A different forecaster has a
different error profile, so it gets its own settings (canids/profile.py). Settings searched (all else in the profile
stays as it is):
  1. threshold policy: window {25, 50, 100} x false-alarm budget {0.03, 0.05, 0.1} alerts/min x yardstick percentile
     {99.5, 99.9, 99.99}. The threshold itself is still fitted from the profile's normal files only; the budget and
     percentile only say how strict that fit is. Chosen by the first halves' average event-level strict F1
     (settings within 0.002 of the best are separated by the best first-half attack naming each can reach, with the plateau
     test's flat_share {0, .3, .5, .7, .9} chosen jointly -- no attack type may lose more than 3 points against the
     original single-tick rule -- then by higher average tick F1; a setting must not buy detection by ruining naming).
  2. alert start/stop refinement (runs.refine_alert_bounds): none, or (alpha, max back, trim) from a small grid, chosen by
     the first halves' average tick F1 while the average event strict F1 stays within 0.01 of unrefined.
Attack labels are used only to score the first halves for this choice (the same protocol as
scripts/detector_split_check.py and boundary_split_check.py); the whole-file numbers of a tuned profile are therefore
optimistic, and only the second-half table is out of sample. Needs a cache from scripts/cache_gru_outputs.py.

Usage:
    python scripts/tune_profile.py <cache_dir> --profile configs/pipeline_tcn.json            # report only
    python scripts/tune_profile.py <cache_dir> --profile configs/pipeline_tcn.json --write configs/pipeline_tcn.json
"""

from __future__ import annotations

import argparse
import dataclasses
from itertools import product
from pathlib import Path

import numpy as np

from canids.cache_eval import ATTACKS, branch2_outputs, fit_from_cache, load_cached, score_attacks, scores_of
from canids.classify import classify_any, type_accuracy, type_confusion
from canids.config import EVENT_ALERT_MERGE_GAP_TICKS, EVENT_MERGE_GAP_TICKS, GRID_STEP_SECONDS
from canids.evaluate import detection_metrics, event_metrics, quiet_split_point
from canids.models.rate_limits import Branch2Model
from canids.profile import PipelineProfile
from canids.relational import RelationalModel
from canids.runs import merge_runs, refine_alert_bounds, runs

WINDOWS = [25, 50, 100]
BUDGETS = [0.03, 0.05, 0.1]
PERCENTILES = [99.5, 99.9, 99.99]
REFINES = [None, (0.75, 100, 25), (0.75, 100, 50), (0.75, 100, 75), (0.9, 100, 50), (0.9, 100, 75)]
TIE = 0.002
NAMING_FLAT_SHARES = [0.0, 0.3, 0.5, 0.7, 0.9]
MAX_TYPE_LOSS = 0.03
EVENT_SLACK = 0.01


def _halves(flags, gt, name, split, refine_args):
    """(tick metrics first, tick second, event first, event second) of `flags` after optional refinement."""
    t = split
    return (
        detection_metrics(flags[:t], gt[:t], name, 0.0), detection_metrics(flags[t:], gt[t:], name, 0.0),
        event_metrics(flags[:t], gt[:t], name, GRID_STEP_SECONDS), event_metrics(flags[t:], gt[t:], name, GRID_STEP_SECONDS),
    )


def _avg(results, idx, attr):
    return float(np.nanmean([getattr(results[a][idx], attr) for a in ATTACKS]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("--profile", type=Path, required=True, help="the profile to start from (its other settings are kept)")
    parser.add_argument("--write", type=Path, default=None, help="save the tuned profile here")
    parser.add_argument("--name", default=None, help="name stored in the tuned profile")
    args = parser.parse_args()

    base = PipelineProfile.load(args.profile)
    relational = RelationalModel.load(Path(base.relational_model))
    b2 = branch2_outputs(Branch2Model.load(Path(base.branch2_model)))
    splits = {a: quiet_split_point(load_cached(args.cache_dir, a)["gt"].astype(bool)) for a in ATTACKS}

    def first_half_naming(outputs, gts, det, prof, flat_share=0.0):
        """Per attack type: (alerts named correctly, alerts) among alerts that start in the first half and overlap an attack."""
        per = {}
        for a in ATTACKS:
            gs, ge = merge_runs(*runs(gts[a]), EVENT_MERGE_GAP_TICKS)
            in_event = np.zeros(len(gts[a]), dtype=bool)
            for x, y in zip(gs, ge):
                in_event[x:y] = True
            ok = tot = 0
            for al in classify_any(
                outputs[a], det, EVENT_ALERT_MERGE_GAP_TICKS, prof.smooth_lookahead_ticks, prof.branch2_min_share,
                flat_share, prof.incident_gap_ticks,
            ):
                if al.start < splits[a] and in_event[al.start:al.end].any():
                    tot += 1
                    ok += al.label == a
            per[a] = (ok, tot)
        return per

    def best_naming(outputs, gts, det, prof):
        """The plateau-test setting (flat_share) with the best first-half naming, provided no attack type loses more than
        MAX_TYPE_LOSS against the original single-tick rule (flat_share 0). Returns (flat_share, overall accuracy)."""
        def overall(per):
            return sum(v[0] for v in per.values()) / max(sum(v[1] for v in per.values()), 1)
        base_per = first_half_naming(outputs, gts, det, prof, 0.0)
        best = (0.0, overall(base_per))
        for fs in NAMING_FLAT_SHARES[1:]:
            per = first_half_naming(outputs, gts, det, prof, fs)
            if all(per[a][0] / max(per[a][1], 1) >= base_per[a][0] / max(base_per[a][1], 1) - MAX_TYPE_LOSS for a in ATTACKS):
                if overall(per) > best[1] + 1e-9:
                    best = (fs, overall(per))
        return best

    def run(profile, refine):
        det = fit_from_cache(args.cache_dir, profile, relational)
        outputs, gts = score_attacks(args.cache_dir, det, b2)
        res = {}
        for a in ATTACKS:
            flags = outputs[a].flags
            if refine is not None:
                flags = refine_alert_bounds(flags, scores_of(outputs[a]), det.tau, refine[0], refine[1], refine[2], EVENT_ALERT_MERGE_GAP_TICKS)
            res[a] = _halves(flags, gts[a], a, splits[a], refine)
        return det, outputs, gts, res

    # ---- 1. threshold policy, chosen on first halves (event strict F1)
    print(f"profile {base.name!r}; first halves only are used for choosing\n")
    print("Step 1 -- threshold policy (no refinement): average over the 5 attacks, FIRST halves")
    print(f"  {'window':>6s} {'budget/min':>10s} {'percentile':>10s} {'tau':>6s} {'event F1':>9s} {'tick F1':>8s} {'false alarms':>12s} {'naming':>7s}")
    table = {}
    for w, b, p in product(WINDOWS, BUDGETS, PERCENTILES):
        prof = dataclasses.replace(base, window=w, target_alerts_per_minute=b, scale_percentile=p)
        det, outs_, gts_, res = run(prof, None)
        # naming is only computed for settings that are within TIE of the best detection so far (it is the tie-breaker);
        # its flat_share setting is chosen jointly with the detector settings
        table[(w, b, p)] = (prof, res)
        del outs_, gts_  # keep memory flat; the few near-ties are re-run below
        fa = sum(res[a][2].n_false_alarms for a in ATTACKS)
        print(f"  {w:6d} {b:10.2f} {p:10.2f} {det.tau:6.2f} {_avg(res, 2, 'strict_f1'):9.3f} {_avg(res, 0, 'f1'):8.3f} {fa:12d}", flush=True)
    best_event = max(_avg(v[1], 2, "strict_f1") for v in table.values())
    near = [k for k, v in table.items() if _avg(v[1], 2, "strict_f1") >= best_event - TIE]
    print(f"\n  detection ties (within {TIE} of the best first-half event F1): {len(near)} settings; the tie is broken by the")
    print("  best first-half attack naming each can reach (flat_share chosen with it), then by tick F1:")
    named = {}
    for k in near:
        prof, res = table[k]
        det, outs_, gts_, _ = run(prof, None)
        fs, acc_ = best_naming(outs_, gts_, det, prof)
        named[k] = (fs, acc_)
        print(f"    window {k[0]:3d} budget {k[1]:.2f} percentile {k[2]:6.2f}: naming {acc_:.3f} with flat_share {fs}", flush=True)
    chosen_key = max(near, key=lambda k: (round(named[k][1], 3), -named[k][0], _avg(table[k][1], 0, "f1")))
    chosen = dataclasses.replace(table[chosen_key][0], plateau_flat_share=named[chosen_key][0])
    print(f"\n  -> chosen from first halves: window={chosen_key[0]}, budget={chosen_key[1]}/min, percentile={chosen_key[2]}, flat_share={chosen.plateau_flat_share}")

    # ---- 2. alert refinement on top of it
    det, outputs, gts, _ = run(chosen, None)
    print("\nStep 2 -- alert start/stop refinement on the chosen settings, FIRST halves")
    ref_res = {}
    for r in REFINES:
        _, _, _, res = run(chosen, r)
        ref_res[r] = res
        print(f"  refine {str(r):>18s}: tick F1 {_avg(res, 0, 'f1'):.3f}  event strict F1 {_avg(res, 2, 'strict_f1'):.3f}", flush=True)
    base_event_f1 = _avg(ref_res[None], 2, "strict_f1")
    ok = [r for r in REFINES if _avg(ref_res[r], 2, "strict_f1") >= base_event_f1 - EVENT_SLACK]
    chosen_refine = max(ok, key=lambda r: _avg(ref_res[r], 0, "f1"))
    print(f"\n  -> chosen refinement: {chosen_refine}")

    # ---- report: second halves, current profile vs tuned
    cur_det, cur_out, cur_gts, cur_res = run(base, base.refine and tuple(base.refine))
    tuned_res = ref_res[chosen_refine]

    def show(label, res):
        print(f"\nSECOND halves (never used for choosing) -- {label}")
        print(f"  {'attack':12s} {'tick P':>7s} {'tick R':>7s} {'tick F1':>8s} | {'events':>6s} {'found':>6s} {'alerts':>6s} {'false':>6s} {'strictF1':>9s}")
        for a in ATTACKS:
            tm, em = res[a][1], res[a][3]
            print(
                f"  {a:12s} {tm.precision:7.3f} {tm.recall:7.3f} {tm.f1:8.3f} | {em.n_events:6d} {em.n_detected:6d} "
                f"{em.n_alerts:6d} {em.n_false_alarms:6d} {em.strict_f1:9.3f}"
            )
        print(
            f"  average: tick P {_avg(res, 1, 'precision'):.3f} R {_avg(res, 1, 'recall'):.3f} F1 {_avg(res, 1, 'f1'):.3f}   "
            f"event strict F1 {_avg(res, 3, 'strict_f1'):.3f}   false alarms {sum(res[a][3].n_false_alarms for a in ATTACKS)}"
        )

    show(f"current settings (window {base.window}, budget {base.target_alerts_per_minute}, percentile {base.scale_percentile}, refine {base.refine})", cur_res)
    show(f"tuned settings (window {chosen.window}, budget {chosen.target_alerts_per_minute}, percentile {chosen.scale_percentile}, refine {chosen_refine})", tuned_res)

    # naming accuracy is not part of the choice; report it so it cannot be missed if tuning hurts it
    print("\nWhole-file attack naming (alerts named correctly), for information:")
    for label, d, o, g, prof in (("current", cur_det, cur_out, cur_gts, base), ("tuned", det, outputs, gts, chosen)):
        counts = type_confusion(
            o, g, d, lookahead_ticks=prof.smooth_lookahead_ticks, min_share=prof.branch2_min_share,
            flat_share=prof.plateau_flat_share, incident_gap_ticks=prof.incident_gap_ticks,
        )
        acc = type_accuracy(counts, list(o))
        ok_n, tot = sum(v[0] for v in acc.values()), sum(v[1] for v in acc.values())
        print(f"  {label:8s} {ok_n}/{tot} = {ok_n / tot:.3f}   " + "  ".join(f"{t} {v[0]}/{v[1]}" for t, v in acc.items()))

    if args.write:
        tuned = dataclasses.replace(
            chosen, name=args.name or f"{base.forecaster}-tuned", refine=list(chosen_refine) if chosen_refine else None,
            notes=(
                f"Detector settings chosen on the FIRST halves of the attack files by scripts/tune_profile.py (window, false-alarm "
                f"budget, yardstick percentile, alert refinement); starting profile: {base.name!r}. Whole-file numbers with this "
                f"profile are optimistic; the second-half numbers in docs/14 are out of sample."
            ),
        )
        tuned.save(args.write)
        print(f"\nwrote {args.write}")


if __name__ == "__main__":
    main()
