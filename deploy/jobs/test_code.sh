#!/bin/bash

# Update environment
pip install -e ".[dev]"
pip install -U "ray[air] @ https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp310-cp310-manylinux2014_x86_64.whl"

# Test code
python -m pytest tests/code --verbose --cov src/madewithml --cov-config=pyproject.toml --cov-report html --disable-warnings > results/test_code_results.txt

# Save to S3
python deploy/utils.py save-to-s3 \
    --file-path results/test_code_results.txt \
    --bucket-name $s3_bucket_name \
    --bucket-path $username/results/$commit_id/test_code_results.txt
