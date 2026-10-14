# Result files

Each file is the FINAL SUMMARY printed by `scripts/run_evaluation.py ... --summary-only` on the real SynCAN attack files
(UTF-8 copies of the repo-root `results*.txt`, which PowerShell wrote as UTF-16; the originals are untouched). Every file
has the same four tables: (1) tick-level confusion counts per attack, (2) event-level results per attack, (3) the
attack-type confusion matrix with per-type and overall accuracy, and (4) from file 05 on, attack naming at tick level.
All results are the **fused** verdict (Branch 1 OR Branch 2). How to read them and how they were produced:
`docs/14-tcn-and-calibration.md`.

| file | forecaster | detector calibrated on | settings | alerts named right |
|---|---|---|---|---|
| `01_gru_old_calibration.txt` | GRU | train_1 val + 10-min train_2 slice | the original (GRU) settings | 571/614 = 93.0% |
| `02_tcn_train1_old_calibration.txt` | TCN (train_1 only) | same 62 min | the GRU's settings | 574/626 = 91.7% (159 false-alarm alerts) |
| `03_gru_long_calibration_untuned.txt` | GRU | train_1 val + 35-min train_2/train_3 slices | the GRU's settings | 572/612 = 93.5% |
| `04_tcn_train1_long_calibration_untuned.txt` | TCN (train_1 only) | same 122 min | the GRU's settings | 561/614 = 91.4% |
| `05_FINAL_gru_tuned.txt` | GRU | same 122 min | `configs/pipeline_gru_tuned.json` | **576/609 = 94.6%** |
| `06_FINAL_tcn_all_tuned.txt` | TCN (all train files) | validation tails of train_1..4 (207 min) | `configs/pipeline_tcn.json` | **571/628 = 90.9%** |
| `07_tcn_all_untuned_partner_run.txt` | TCN (all train files) | validation tails of train_1..4 | the GRU's settings | 530/603 = 87.9% (138 false-alarm alerts) |
| `08_gru_tails_calibration_partner_run.txt` | GRU | validation tails of train_1..4 | the GRU's settings | 573/610 = 93.9% |

Files 07 and 08 are the partner's runs on their GPU machine (`logs/eval_*_all.log`); the rest were run on the main machine.
Files 05 and 06 are the headline results. Their settings were chosen on the first halves of the attack files, so their
whole-file numbers are optimistic; the out-of-sample (second-half) numbers are in `docs/14-tcn-and-calibration.md`.
