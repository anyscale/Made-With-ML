#!/bin/bash

# Test code
python -m pytest tests/code --verbose --disable-warnings > test_code_results.txt

# Save to S3
RESULTS_FILE=test_code_results.txt
python deploy/utils.py save-to-s3 \
    --file-path $RESULTS_FILE \
    --bucket-name $s3_bucket_name \
    --bucket-path $username/results/$commit_id/$RESULTS_FILE
