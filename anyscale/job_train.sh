#!/bin/bash

DATASET_LOC="https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/madewithml/dataset.csv"
HOLDOUT_LOC="https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/madewithml/holdout.csv"
TRAIN_LOOP_CONFIG="{'dropout_p': 0.5, 'lr': 1e-4, 'lr_factor': 0.8, 'lr_patience': 3}"

python ./src/madewithml/train.py llm \
    "$DATASET_LOC" \
    "$TRAIN_LOOP_CONFIG" \
    --use-gpu \
    --num-cpu-workers 40 \
    --num-gpu-workers 2 \
    --num-epochs 10 \
    --batch-size 256
