#!/bin/bash

# Train model
set -xe
git pull
EXPERIMENT_NAME="llm"
DATASET_LOC="https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/madewithml/dataset.csv"
TRAIN_LOOP_CONFIG='{"dropout_p": 0.5, "lr": 1e-4, "lr_factor": 0.8, "lr_patience": 3}'
python src/madewithml/train.py \
    "$EXPERIMENT_NAME" \
    "$DATASET_LOC" \
    "$TRAIN_LOOP_CONFIG" \
    --use-gpu \
    --num-cpu-workers 40 \
    --num-gpu-workers 2 \
    --num-epochs 10 \
    --batch-size 256 \
    --results-fp ./training_result.json

# Workaround for /mnt/user_storage: Upload mlflow to S3
MODEL_REGISTRY=$(python -c "from madewithml.config import MODEL_REGISTRY; print(str(MODEL_REGISTRY))")
aws s3 sync "$MODEL_REGISTRY" s3://goku-mlops/mlflow

# Print training results
set +x
echo "####TRAIN_OUT####"
cat ./training_result.json
echo "####TRAIN_END####"
