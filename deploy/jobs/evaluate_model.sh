#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$PWD

# Workaround for /mnt/user_storage: Download mlflow from S3
MODEL_REGISTRY=$(python -c "from madewithml.config import MODEL_REGISTRY; print(str(MODEL_REGISTRY))")
aws s3 sync $S3_BUCKET/$GITHUB_USERNAME/mlflow "$MODEL_REGISTRY"

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
    --results-fp $RESULTS_FILE

# Save to S3
python deploy/utils.py save-to-s3 \
    --file-path $RESULTS_FILE \
    --s3-path $S3_BUCKET/$GITHUB_USERNAME/pull_requests/$PR_NUM/commits/$COMMIT_ID/$RESULTS_FILE
