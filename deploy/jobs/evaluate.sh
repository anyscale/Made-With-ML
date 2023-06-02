#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$PWD

# Get run ID
if [[ -z "${RUN_ID}" ]]; then  # if RUN_ID is set use it, else get the best run
    RUN_ID=$(python madewithml/predict.py get-best-run-id --experiment-name $EXPERIMENT_NAME --metric val_loss --mode ASC)
fi

# Evaluate
export RESULTS_FILE=evaluation_results.json
export HOLDOUT_LOC="https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/madewithml/holdout.csv"
python madewithml/evaluate.py \
    --run-id $RUN_ID \
    --dataset-loc $HOLDOUT_LOC \
    --num-cpu-workers 2 \
    --results-fp $RESULTS_FILE

# Save to S3
python deploy/utils.py save-to-s3 \
    --file-path $RESULTS_FILE \
    --bucket-name $S3_BUCKET_NAME \
    --bucket-path $GITHUB_USERNAME/pull_requests/$PR_NUM/commits/$COMMIT_ID/$RESULTS_FILE
