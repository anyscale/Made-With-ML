#!/bin/bash
python3 -m pip install ".[dev]"  # workaround to update madewithml package (but only for head node)

# Test code
RESULTS_FILE=test_code_results.txt
python -m pytest tests/code --verbose --disable-warnings > $RESULTS_FILE

# Save to S3
python deploy/utils.py save-to-s3 \
    --file-path $RESULTS_FILE \
    --bucket-name $s3_bucket_name \
    --bucket-path $username/results/$commit_id/$RESULTS_FILE
