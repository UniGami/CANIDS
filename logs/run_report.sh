#!/bin/bash
cd /c/Users/sreen/CANIDS
until grep -q GRU_FINAL_DONE logs/eval_gru_final.log; do sleep 10; done
/c/Users/sreen/anaconda3/envs/canids/python.exe -u scripts/report_tables.py --gru logs/sweep/gru.npz --tcn logs/sweep/tcn.npz --tcn-floor 0.010 > logs/report_tables.log 2>&1
echo REPORT_DONE >> logs/report_tables.log
