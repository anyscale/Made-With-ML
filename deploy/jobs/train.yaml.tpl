name: train-model
project_id: PROJECT_ID
compute_config: COMPUTE_CONFIG_NAME
build_id: CLUSTER_ENV_ID
runtime_env:
  working_dir: .
  upload_path: UPLOAD_PATH
entrypoint: bash deploy/jobs/train.sh
max_retries: 0
