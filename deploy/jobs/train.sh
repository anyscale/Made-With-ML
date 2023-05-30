#!/bin/bash

# Update environment
pip install -e ".[dev]"
pip install -U "ray[air] @ https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp310-cp310-manylinux2014_x86_64.whl"

# Get run ID
if [[ -z "${run_id}" ]]; then  # if RUN_ID is set use it, else get the best run
    run_id=$(python src/madewithml/predict.py get-best-run-id --experiment-name $experiment_name --metric val_loss --mode ASC)
fi

# Train
DATASET_LOC="https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/madewithml/dataset.csv"
TRAIN_LOOP_CONFIG='{"dropout_p": 0.5, "lr": 1e-4, "lr_factor": 0.8, "lr_patience": 3}'
python src/madewithml/train.py \
    "$experiment_name" \
    "$DATASET_LOC" \
    "$TRAIN_LOOP_CONFIG" \
    --use-gpu \
    --num-cpu-workers 10 \
    --num-gpu-workers 2 \
    --num-epochs 10 \
    --batch-size 256 \
    --results-fp training_results.json

# Save to S3
python deploy/utils.py save-to-s3 \
    --file-path training_results.json \
    --bucket-name $s3_bucket_name \
    --bucket-path $username/results/$commit_id/training_results.json
