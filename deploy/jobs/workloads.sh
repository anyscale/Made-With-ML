#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$PWD

# Test data
export DATASET_LOC="https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/madewithml/dataset.csv"
pytest --dataset-loc=$DATASET_LOC tests/data --verbose --disable-warnings

# Test code
python -m pytest tests/code --verbose --disable-warnings

# Train
export EXPERIMENT_NAME=llm
export DATASET_LOC="https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/madewithml/dataset.csv"
export TRAIN_LOOP_CONFIG='{"dropout_p": 0.5, "lr": 1e-4, "lr_factor": 0.8, "lr_patience": 3}'
export RESULTS_FILE=training_results.json
python madewithml/train.py \
    --experiment-name "$EXPERIMENT_NAME" \
    --dataset-loc "$DATASET_LOC" \
    --train-loop-config "$TRAIN_LOOP_CONFIG" \
    --num-workers 2 \
    --cpu-per-worker 10 \
    --gpu-per-worker 1 \
    --num-epochs 10 \
    --batch-size 256 \
    --results-fp $RESULTS_FILE

# Get and save run ID
export RUN_ID=$(python -c "import os; from madewithml import utils; d = utils.load_dict(os.getenv('RESULTS_FILE')); print(d['run_id'])")
export RUN_ID_FILE=/mnt/user_storage/run_id.txt
echo $RUN_ID > $RUN_ID_FILE  # used for serving later

# Evaluate
export HOLDOUT_LOC="https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/madewithml/holdout.csv"
export RESULTS_FILE=evaluation_results.json
python madewithml/evaluate.py \
    --run-id $RUN_ID \
    --dataset-loc $HOLDOUT_LOC \
    --results-fp $RESULTS_FILE

# Test model
pytest --run-id=$RUN_ID tests/model --verbose --disable-warnings
