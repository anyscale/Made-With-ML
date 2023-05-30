#!/bin/bash

# Update environment
pip install -e ".[dev]"
pip install -U "ray[air] @ https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp310-cp310-manylinux2014_x86_64.whl"

# Get run ID
if [[ -z "${run_id}" ]]; then  # if RUN_ID is set use it, else get the best run
    run_id=$(python src/madewithml/predict.py get-best-run-id --experiment-name $experiment_name --metric val_loss --mode ASC)
fi

# Evaluate
HOLDOUT_LOC="https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/madewithml/holdout.csv"
python src/madewithml/evaluate.py \
    --run-id $run_id \
    --dataset-loc $HOLDOUT_LOC \
    --num-cpu-workers 2 \
    --results-fp evaluation_results.json

# Save to S3
python deploy/utils.py save-to-s3 \
    --file-path evaluation_results.json \
    --bucket-name $s3_bucket_name \
    --bucket-path $username/results/$commit_id/evaluation_results.json
