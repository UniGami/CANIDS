from canids.evaluate import DetectionMetrics, EventMetrics
from canids.report import format_final_summary


def _tick(name, tp, fp, fn, n=1000):
    p, r = tp / (tp + fp), tp / (tp + fn)
    return DetectionMetrics(name, 99.5, n, tp + fn, tp + fp, tp, fp, fn, p, r, 2 * p * r / (p + r))


def _event(name, events, found, alerts, false):
    return EventMetrics(
        attack_type=name, n_events=events, n_detected=found, n_alerts=alerts, n_true_alerts=alerts - false,
        n_false_alarms=false, n_pure_alerts=alerts - false, mean_coverage=0.9, false_alarms_per_minute=0.1,
        precision=(alerts - false) / alerts, strict_precision=(alerts - false) / alerts, recall=found / events, f1=0.9, strict_f1=0.9,
    )


def test_summary_contains_the_confusion_counts_and_overall_numbers():
    ticks = [_tick("plateau", 80, 20, 20), _tick("drift", 60, 10, 40)]
    events = [_event("plateau", 10, 9, 12, 2), _event("drift", 10, 8, 10, 1)]
    counts = {("plateau", "plateau"): 8, ("plateau", "replay"): 2, ("drift", "drift"): 6, ("drift", "replay"): 3, ("false alarm", "replay"): 4}
    text = format_final_summary(ticks, events, counts, "test verdict")

    assert "FINAL SUMMARY -- test verdict" in text
    # TN = 1000 - 80 - 20 - 20 = 880 for plateau; sums: TP 140, FP 30, TN 880 + 890, FN 60
    assert "880" in text and "890" in text and "ALL FILES" in text
    assert "140" in text and "1770" in text  # total TN
    assert "17 of 22 attacks found" not in text and "17 of 20 attacks found = 85.0%" in text
    assert "14 of 19 attack alerts named correctly = 73.7%" in text  # (8 + 6) / (10 + 9)
    assert "false alarm" in text and "accuracy*" in text


def test_summary_handles_attacks_with_no_alerts_without_crashing():
    ticks = [_tick("plateau", 80, 20, 20)]
    events = [_event("plateau", 10, 9, 12, 2)]
    text = format_final_summary(ticks, events, {}, "empty")
    assert "0 of 0 attack alerts" in text
