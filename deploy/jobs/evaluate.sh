#!/bin/bash

# Evaluate model
set -xe
if [[ -z "${run_id}" ]]; then  # if RUN_ID is set use it, else get the best run
    run_id=$(python src/madewithml/predict.py get-best-run-id --experiment-name $experiment_name --metric val_loss --mode ASC)
fi
HOLDOUT_LOC="https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/madewithml/holdout.csv"
python src/madewithml/evaluate.py \
    --run-id $run_id \
    --dataset-loc $HOLDOUT_LOC \
    --num-cpu-workers 2 \
    --results-fp ./evaluation_results.json

# Save to S3
python src/deploy/utils/utils.py save-to-s3 \
    --file ./evaluation_results.json \
    --bucket $s3_bucket \
    --path evaluation_results-$job_name.json
