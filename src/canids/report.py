"""The one-screen FINAL SUMMARY printed at the very end of scripts/run_evaluation.py:
for every attack, the tick-level confusion counts and the event-level results, then the
attack-TYPE confusion matrix with per-type and overall accuracy.
"""

from __future__ import annotations

import math

from canids.classify import (
    DRIFT,
    FALSE_ALARM,
    FLOODING,
    FUZZING,
    PLATEAU,
    REPLAY,
    SUPPRESSION,
    type_accuracy,
)
from canids.evaluate import DetectionMetrics, EventMetrics

TYPE_COLUMNS = [SUPPRESSION, PLATEAU, DRIFT, REPLAY, FLOODING, FUZZING]


def _f(x: float, width: int = 6) -> str:
    return f"{'n/a':>{width}s}" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:{width}.3f}"


def _ratio(num: float, den: float) -> float:
    return num / den if den > 0 else float("nan")


def format_final_summary(
    tick_metrics: list[DetectionMetrics],
    event_metrics: list[EventMetrics],
    type_counts: dict[tuple[str, str], int],
    title: str,
    tick_naming: dict[str, tuple[int, int, int, int]] | None = None,
) -> str:
    """`tick_metrics` / `event_metrics` / `type_counts` are one verdict's results (e.g. the fused one)."""
    out: list[str] = []
    bar = "=" * 100
    out += [bar, f"FINAL SUMMARY -- {title}", bar]

    # 1. tick-level confusion counts
    out += ["", "1) TICK LEVEL (one tick = 0.01 s): confusion counts per attack file", ""]
    out.append(f"   {'attack':13s} {'TP':>9s} {'FP':>9s} {'TN':>9s} {'FN':>9s}   {'precision':>9s} {'recall':>7s} {'F1':>6s} {'accuracy*':>10s}")
    tp = fp = tn = fn = 0
    for m in tick_metrics:
        t_n = m.n_ticks - m.true_positives - m.false_positives - m.false_negatives
        tp, fp, tn, fn = tp + m.true_positives, fp + m.false_positives, tn + t_n, fn + m.false_negatives
        acc = _ratio(m.true_positives + t_n, m.n_ticks)
        out.append(
            f"   {m.attack_type:13s} {m.true_positives:9d} {m.false_positives:9d} {t_n:9d} {m.false_negatives:9d}   "
            f"{_f(m.precision, 9)} {_f(m.recall, 7)} {_f(m.f1)} {_f(acc, 10)}"
        )
    p, r = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    f1 = _ratio(2 * p * r, p + r) if not (math.isnan(p) or math.isnan(r)) else float("nan")
    out.append(f"   {'-' * 98}")
    out.append(
        f"   {'ALL FILES':13s} {tp:9d} {fp:9d} {tn:9d} {fn:9d}   {_f(p, 9)} {_f(r, 7)} {_f(f1)} {_f(_ratio(tp + tn, tp + fp + tn + fn), 10)}"
    )
    out.append(
        "   * accuracy = (TP+TN)/all ticks. About 85% of ticks are normal, so it flatters any detector that flags little;"
        "\n     read precision / recall / F1 for how well the attacks themselves are found."
    )

    # 2. event level
    out += ["", "2) EVENT LEVEL: each attack interval and each alert counts once", ""]
    out.append(
        f"   {'attack':13s} {'attacks':>8s} {'found':>6s} {'alerts':>7s} {'false':>6s} {'FA/min':>7s}   {'strict prec':>11s} {'recall':>7s} {'strict F1':>10s}"
    )
    ev, found, alerts, false_alarms = 0, 0, 0, 0
    for m in event_metrics:
        ev, found, alerts, false_alarms = ev + m.n_events, found + m.n_detected, alerts + m.n_alerts, false_alarms + m.n_false_alarms
        out.append(
            f"   {m.attack_type:13s} {m.n_events:8d} {m.n_detected:6d} {m.n_alerts:7d} {m.n_false_alarms:6d} {_f(m.false_alarms_per_minute, 7)}   "
            f"{_f(m.strict_precision, 11)} {_f(m.recall, 7)} {_f(m.strict_f1, 10)}"
        )
    out.append(f"   {'-' * 98}")
    out.append(
        f"   {'ALL FILES':13s} {ev:8d} {found:6d} {alerts:7d} {false_alarms:6d}   -> {found} of {ev} attacks found "
        f"= {_ratio(found, ev):.1%} ; {alerts - false_alarms} of {alerts} alerts overlap an attack"
    )

    # 3. attack-type confusion matrix
    attacks = [m.attack_type for m in tick_metrics]
    cols = [c for c in TYPE_COLUMNS if any(counts for (_, label), counts in type_counts.items() if label == c)]
    cols = cols or TYPE_COLUMNS[:4]
    out += ["", "3) ATTACK-TYPE CONFUSION MATRIX: rows = what really happened, columns = what the system said", "   (counts alerts that overlap an attack; the last row is alerts that overlap none)", ""]
    out.append(f"   {'real attack':14s} " + " ".join(f"{c:>12s}" for c in cols) + f" {'correct':>10s} {'accuracy':>9s}")
    acc = type_accuracy(type_counts, attacks)
    ok_all = total_all = 0
    for a in attacks:
        ok, total = acc[a]
        ok_all, total_all = ok_all + ok, total_all + total
        out.append(
            f"   {a:14s} " + " ".join(f"{type_counts.get((a, c), 0):12d}" for c in cols) + f" {f'{ok}/{total}':>10s} {_f(_ratio(ok, total), 9)}"
        )
    out.append("   " + "-" * (14 + 13 * len(cols) + 21))
    out.append(f"   {FALSE_ALARM:14s} " + " ".join(f"{type_counts.get((FALSE_ALARM, c), 0):12d}" for c in cols))
    out.append("")
    out.append(
        f"   OVERALL TYPE ACCURACY (whole dataset): {ok_all} of {total_all} attack alerts named correctly = {_ratio(ok_all, total_all):.1%}"
    )
    if tick_naming:
        out.append("")
        out.append("4) ATTACK-TYPE ACCURACY AT TICK LEVEL: of each attack's own ticks, how many sit inside an alert named with")
        out.append("   the right type / a wrong type / no alert at all")
        out.append("")
        out.append(
            f"   {'attack':14s} {'attack ticks':>12s} {'right name':>11s} {'wrong name':>11s} {'not flagged':>12s} {'named right, of flagged':>24s}"
        )
        tot = [0, 0, 0, 0]
        for a, (n, r, w, m) in tick_naming.items():
            tot = [tot[0] + n, tot[1] + r, tot[2] + w, tot[3] + m]
            out.append(
                f"   {a:14s} {n:12d} {_f(_ratio(r, n), 11)} {_f(_ratio(w, n), 11)} {_f(_ratio(m, n), 12)} {_f(_ratio(r, r + w), 24)}"
            )
        out.append("   " + "-" * 78)
        out.append(
            f"   {'ALL FILES':14s} {tot[0]:12d} {_f(_ratio(tot[1], tot[0]), 11)} {_f(_ratio(tot[2], tot[0]), 11)} "
            f"{_f(_ratio(tot[3], tot[0]), 12)} {_f(_ratio(tot[1], tot[1] + tot[2]), 24)}"
        )
    return "\n".join(out)
