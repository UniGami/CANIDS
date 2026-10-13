#!/bin/bash
cd /c/Users/sreen/CANIDS
until [ -f logs/diag/tcn.npz ] || grep -q Traceback logs/diag_collect_tcn.log; do sleep 10; done
sleep 5
/c/Users/sreen/anaconda3/envs/canids/python.exe -u scripts/diagnose_false_alerts.py collect --forecaster gru --model-path models/gru_syncan_train1.pt --out logs/diag/gru.npz > logs/diag_collect_gru.log 2>&1
echo GRU_COLLECT_DONE >> logs/diag_collect_gru.log
