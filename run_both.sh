#!/usr/bin/env bash
set -e

CUDA_VISIBLE_DEVICES=0 python3 model_training_trial1.py > logs_classifier.txt 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=1 python3 model_training_trial2.py > logs_center.txt 2>&1 &
PID2=$!

#wait $PID1
wait $PID2