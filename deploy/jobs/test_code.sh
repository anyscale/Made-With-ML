#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$PWD

# Test code
export RESULTS_FILE=test_code_results.txt
python -m pytest tests/code --verbose --disable-warnings > $RESULTS_FILE

# Save to S3
python deploy/utils.py save-to-s3 \
    --file-path $RESULTS_FILE \
    --bucket-name $S3_BUCKET_NAME \
    --bucket-path $GITHUB_USERNAME/pull_requests/$PR_NUM/commits/$COMMIT_ID/$RESULTS_FILE
