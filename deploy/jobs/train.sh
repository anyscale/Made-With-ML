#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$PWD

# Train
export RESULTS_FILE=training_results.json
export DATASET_LOC="https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/madewithml/dataset.csv"
export TRAIN_LOOP_CONFIG='{"dropout_p": 0.5, "lr": 1e-4, "lr_factor": 0.8, "lr_patience": 3}'
python madewithml/train.py \
    "$EXPERIMENT_NAME" \
    "$DATASET_LOC" \
    "$TRAIN_LOOP_CONFIG" \
    --use-gpu \
    --num-cpu-workers 10 \
    --num-gpu-workers 2 \
    --num-epochs 10 \
    --batch-size 256 \
    --results-fp $RESULTS_FILE

# Save to S3
python deploy/utils.py save-to-s3 \
    --file-path $RESULTS_FILE \
    --s3-path $S3_BUCKET/$GITHUB_USERNAME/pull_requests/$PR_NUM/commits/$COMMIT_ID/$RESULTS_FILE

# Workaround for /mnt/user_storage: Upload mlflow to S3
MODEL_REGISTRY=$(python -c "from madewithml.config import MODEL_REGISTRY; print(str(MODEL_REGISTRY))")
aws s3 sync "$MODEL_REGISTRY" $S3_BUCKET/$GITHUB_USERNAME/mlflow
