#!/bin/bash
cd /c/Users/sreen/CANIDS
PY=/c/Users/sreen/anaconda3/envs/canids/python.exe
$PY -u scripts/run_evaluation.py --attack-source syncan --normal-csv data/raw/syncan_train_1.csv --model-type tcn --model-path models/tcn_syncan_train1.pt --batch-size 1024 --skip-residual-report > logs/eval_tcn_t2.log 2>&1
echo "TCN_DONE" >> logs/eval_tcn_t2.log
$PY -u scripts/run_evaluation.py --attack-source syncan --normal-csv data/raw/syncan_train_1.csv --model-path models/gru_syncan_train1.pt --batch-size 1024 --skip-residual-report > logs/eval_gru_t2.log 2>&1
echo "GRU_DONE" >> logs/eval_gru_t2.log
