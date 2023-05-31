#!/bin/bash
python3 -m pip install ".[dev]"  # workaround to update madewithml package (but only for head node)

# Get run ID
if [[ -z "${run_id}" ]]; then  # if RUN_ID is set use it, else get the best run
    run_id=$(python src/madewithml/predict.py get-best-run-id --experiment-name $experiment_name --metric val_loss --mode ASC)
fi

# Train
RESULTS_FILE=training_results.json
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
    --results-fp $RESULTS_FILE

# Save to S3
python deploy/utils.py save-to-s3 \
    --file-path $RESULTS_FILE \
    --bucket-name $s3_bucket_name \
    --bucket-path $username/results/$commit_id/$RESULTS_FILE
