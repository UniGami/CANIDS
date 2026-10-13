#!/bin/bash
cd /c/Users/sreen/CANIDS
PY=/c/Users/sreen/anaconda3/envs/canids/python.exe
EXTRA="data/processed/syncan_train_2_val.csv data/processed/syncan_train_3_val.csv data/processed/syncan_train_4_val.csv"
COMMON="--attack-source syncan --normal-csv data/raw/syncan_train_1.csv --batch-size 1024 --skip-residual-report --detector-extra-normal-csv $EXTRA"
$PY -u scripts/run_evaluation.py $COMMON --forecaster tcn --model-path models/tcn_syncan_all.pt --device cuda --detector-signed-scale-floor 0.01 > logs/eval_tcn_final.log 2>&1
echo "TCN_FINAL_DONE" >> logs/eval_tcn_final.log
$PY -u scripts/run_evaluation.py $COMMON --model-path models/gru_syncan_train1.pt > logs/eval_gru_final.log 2>&1
echo "GRU_FINAL_DONE" >> logs/eval_gru_final.log
