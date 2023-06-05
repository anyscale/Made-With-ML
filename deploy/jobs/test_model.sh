#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$PWD

# Get run ID
if [[ -z "${RUN_ID}" ]]; then  # if RUN_ID is set use it, else get the best run
    RUN_ID=$(python madewithml/predict.py get-best-run-id --experiment-name $EXPERIMENT_NAME --metric val_loss --mode ASC)
fi

# Test model
RESULTS_FILE=test_model_results.txt
pytest --run-id=$RUN_ID tests/model --verbose --disable-warnings > $RESULTS_FILE
cat $RESULTS_FILE

# Save to S3
python deploy/utils.py save-to-s3 \
    --file-path $RESULTS_FILE \
    --s3-path $S3_BUCKET/$GITHUB_USERNAME/pull_requests/$PR_NUM/commits/$COMMIT_ID/$RESULTS_FILE
