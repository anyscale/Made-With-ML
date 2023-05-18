#!/bin/bash

DATASET_LOC="https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/madewithml/dataset.csv"
HOLDOUT_LOC="https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/madewithml/holdout.csv"
TRAIN_LOOP_CONFIG='{"dropout_p": 0.5, "lr": 1e-4, "lr_factor": 0.8, "lr_patience": 3}'

# Workaround for /mnt/user_storage: Download mlflow from S3
MODEL_REGISTRY=$(python -c "from madewithml.config import MODEL_REGISTRY; print(str(MODEL_REGISTRY))")
aws s3 sync s3://kf-mlops-dev/mlflow "$MODEL_REGISTRY"

RUN_ID=$1
python ./src/madewithml/evaluate.py \
    --dataset-loc $HOLDOUT_LOC \
    --num-cpu-workers 2 \
    --run-id $RUN_ID \
    --results-fp ./eval_result.json

# Print best run ID
echo "####EVAL_OUT####"
cat ./eval_result.json
echo "####EVAL_END####"
