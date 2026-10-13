"""Out-of-sample check of the attack-NAMING settings (classify.py), per forecaster profile.

Two naming settings are tried on top of a profile's detector (the detector, its threshold and its alerts do not change):
  flat_share -- rule 2b ("the top-error signal is flat") must hold for this share of the alert's ticks, instead of at
                the single tick where the forecast error peaks (0 = that original rule);
  incident_gap_ticks -- alerts closer together than this are one incident, named from ALL its evidence, and every
                fragment gets that name (None = each alert named on its own).
Like scripts/detector_split_check.py: every attack file is cut in two at a quiet spot; the settings are CHOSEN from the
first halves only (highest share of first-half alerts named correctly, provided no attack type loses more than 3
points against the current rule) and REPORTED on the second halves. Alerts are assigned to a half by where they start.

Usage:
    python scripts/naming_split_check.py <cache_dir> --profile configs/pipeline_tcn.json [--write configs/pipeline_tcn.json]
"""

from __future__ import annotations

import argparse
import dataclasses
from itertools import product
from pathlib import Path

import numpy as np

from canids.cache_eval import ATTACKS, branch2_outputs, fit_from_cache, load_cached, score_attacks
from canids.classify import classify_any
from canids.config import EVENT_ALERT_MERGE_GAP_TICKS, EVENT_MERGE_GAP_TICKS
from canids.evaluate import quiet_split_point
from canids.models.rate_limits import Branch2Model
from canids.profile import PipelineProfile
from canids.relational import RelationalModel
from canids.runs import merge_runs, runs

FLAT_SHARES = [0.0, 0.3, 0.5, 0.7, 0.9]
INCIDENT_GAPS = [None, 100, 300, 600, 1000]
MAX_TYPE_LOSS = 0.03


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--write", type=Path, default=None, help="save the profile with the chosen naming settings here")
    args = parser.parse_args()

    profile = PipelineProfile.load(args.profile)
    relational = RelationalModel.load(Path(profile.relational_model))
    det = fit_from_cache(args.cache_dir, profile, relational)
    outputs, gts = score_attacks(args.cache_dir, det, branch2_outputs(Branch2Model.load(Path(profile.branch2_model))))
    splits = {a: quiet_split_point(gts[a]) for a in ATTACKS}
    in_event = {}
    for a in ATTACKS:
        gs, ge = merge_runs(*runs(gts[a]), EVENT_MERGE_GAP_TICKS)
        m = np.zeros(len(gts[a]), dtype=bool)
        for x, y in zip(gs, ge):
            m[x:y] = True
        in_event[a] = m

    def score(flat_share, gap):
        """{attack: [first-half (right, total), second-half (right, total)]} over alerts that overlap an attack."""
        res = {a: [[0, 0], [0, 0]] for a in ATTACKS}
        for a in ATTACKS:
            for al in classify_any(
                outputs[a], det, EVENT_ALERT_MERGE_GAP_TICKS, profile.smooth_lookahead_ticks, profile.branch2_min_share,
                flat_share, gap,
            ):
                if not in_event[a][al.start : al.end].any():
                    continue
                half = 0 if al.start < splits[a] else 1
                res[a][half][1] += 1
                res[a][half][0] += al.label == a
        return res

    def acc(res, half, types=ATTACKS):
        ok = sum(res[a][half][0] for a in types)
        tot = sum(res[a][half][1] for a in types)
        return ok / max(tot, 1)

    configs = list(product(FLAT_SHARES, INCIDENT_GAPS))
    results = {c: score(*c) for c in configs}
    current = (profile.plateau_flat_share, profile.incident_gap_ticks)
    base = results[(0.0, None)]
    print(f"profile {profile.name!r}; current naming settings (flat_share, incident_gap) = {current}\n")
    print("Step 1 -- choose using ONLY first halves: share of alerts named correctly (per-type loss vs current must be <= 3 points)")
    print(f"  {'flat_share':>10s} {'incident gap':>12s} {'1st half':>9s}  worst type change")
    allowed = []
    for c in configs:
        losses = [acc(results[c], 0, [a]) - acc(base, 0, [a]) for a in ATTACKS if base[a][0][1]]
        ok = min(losses) >= -MAX_TYPE_LOSS
        if ok:
            allowed.append(c)
        print(f"  {c[0]:10.1f} {str(c[1]):>12s} {acc(results[c], 0):9.3f}  {min(losses):+.3f}{'' if ok else '   (a type lost too much)'}")
    # prefer the simplest of the best (fewest changes from the original rule) on ties
    best = max(acc(results[c], 0) for c in allowed)
    chosen = min((c for c in allowed if acc(results[c], 0) >= best - 1e-9), key=lambda c: (c[0] > 0) + (c[1] is not None))
    print(f"\n  -> chosen from first halves: flat_share={chosen[0]}, incident_gap_ticks={chosen[1]}")

    def show(label, res):
        print(f"\nSECOND halves (never used for choosing) -- {label}")
        for a in ATTACKS:
            r, t = res[a][1]
            print(f"  {a:12s} {r:4d}/{t:<4d} = {r / max(t, 1):.3f}")
        print(f"  {'overall':12s} {sum(res[a][1][0] for a in ATTACKS)}/{sum(res[a][1][1] for a in ATTACKS)} = {acc(res, 1):.3f}")

    show("current rule (flat_share 0, no incident grouping)", base)
    show(f"chosen settings flat_share={chosen[0]}, incident_gap={chosen[1]}", results[chosen])
    whole = lambda res: sum(res[a][h][0] for a in ATTACKS for h in (0, 1)) / sum(res[a][h][1] for a in ATTACKS for h in (0, 1))
    print(f"\nwhole files (optimistic: the first halves chose the settings): current {whole(base):.3f} -> chosen {whole(results[chosen]):.3f}")

    if args.write:
        tuned = dataclasses.replace(
            profile, plateau_flat_share=chosen[0], incident_gap_ticks=chosen[1],
            notes=profile.notes + f" Naming settings (flat_share {chosen[0]}, incident gap {chosen[1]}) chosen on first halves by scripts/naming_split_check.py.",
        )
        tuned.save(args.write)
        print(f"\nwrote {args.write}")


if __name__ == "__main__":
    main()
