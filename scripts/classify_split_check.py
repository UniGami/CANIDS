"""First-half / second-half check of the attack-TYPE labels (classify.py).

The type rules have no fitted numbers -- every cutoff is "bigger than normal
traffic ever gets" or the detector's own window length -- but the choice of
WHICH rules to use was made while looking at the attack files. So, as with
scripts/detector_split_check.py, each attack file is cut in half by time (at a
quiet spot) and the labels are reported separately on each half. Similar
numbers on both halves mean the rules are not tuned to particular attacks
(they cannot show the rules carry over to a different car or dataset).

Needs the cache from scripts/cache_gru_outputs.py.

Usage:
    python scripts/classify_split_check.py <cache_dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from canids.classify import DRIFT, FALSE_ALARM, PLATEAU, REPLAY, SUPPRESSION, type_accuracy, type_confusion
from canids.detector import NormalStream, detect, fit_detector
from canids.evaluate import quiet_split_point

ATTACKS = ["plateau", "drift", "replay", "suppression", "flooding"]
LABELS = [SUPPRESSION, PLATEAU, DRIFT, REPLAY]


def _load(cache: Path, name: str) -> dict[str, np.ndarray]:
    d = np.load(cache / f"{name}.npz")
    return {k: d[k] for k in d.files}


def _stream(d: dict[str, np.ndarray]) -> NormalStream:
    return NormalStream(d["residuals"].astype(float), d["staleness"].astype(float), d["values"].astype(float))


def _show(title: str, counts: dict[tuple[str, str], int]) -> None:
    print(f"\n{title}")
    print(f"  {'real attack':14s} " + " ".join(f"{c:>12s}" for c in LABELS))
    for row in [*ATTACKS, FALSE_ALARM]:
        print(f"  {row:14s} " + " ".join(f"{counts.get((row, c), 0):12d}" for c in LABELS))
    acc = type_accuracy(counts, ATTACKS)
    parts = [f"{t} {ok}/{n}" for t, (ok, n) in acc.items() if t != "flooding" and n]
    ok = sum(acc[t][0] for t in ATTACKS if t != "flooding")
    n = sum(acc[t][1] for t in ATTACKS if t != "flooding")
    print(f"  correct: {', '.join(parts)}  ->  overall (excluding flooding) {ok}/{n} = {ok / n:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cache_dir", type=Path)
    args = parser.parse_args()

    det = fit_detector([_stream(_load(args.cache_dir, n)) for n in ("train1_val", "train2")])
    first_out, second_out, first_gt, second_gt = {}, {}, {}, {}
    for a in ATTACKS:
        d = _load(args.cache_dir, a)
        out = detect(det, d["residuals"].astype(float), d["staleness"].astype(float), d["values"].astype(float))
        t = quiet_split_point(d["gt"])
        first_out[a] = _slice(out, slice(0, t))
        second_out[a] = _slice(out, slice(t, None))
        first_gt[a], second_gt[a] = d["gt"][:t], d["gt"][t:]

    _show("FIRST halves", type_confusion(first_out, first_gt, det))
    _show("SECOND halves", type_confusion(second_out, second_gt, det))


def _slice(out, sl):
    from dataclasses import replace

    return replace(
        out,
        score=out.score[sl],
        flags=out.flags[sl],
        channel_scores={k: v[sl] for k, v in out.channel_scores.items()},
        top_signal=out.top_signal[sl],
        top_signal_streak=out.top_signal_streak[sl],
        jump_score=out.jump_score[sl],
    )


if __name__ == "__main__":
    main()
