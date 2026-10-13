#!/bin/bash
cd /c/Users/sreen/CANIDS
PY=/c/Users/sreen/anaconda3/envs/canids/python.exe
$PY -u scripts/sweep_signed_floor.py collect --forecaster tcn --model-path models/tcn_syncan_all.pt --device cuda --out logs/sweep/tcn.npz > logs/sweep_collect_tcn.log 2>&1
echo TCN_COLLECT_DONE >> logs/sweep_collect_tcn.log
$PY -u scripts/sweep_signed_floor.py collect --forecaster gru --model-path models/gru_syncan_train1.pt --out logs/sweep/gru.npz > logs/sweep_collect_gru.log 2>&1
echo GRU_COLLECT_DONE >> logs/sweep_collect_gru.log
