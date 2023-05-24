name: service-mlops
project_id: $PROJECT_ID
compute_config: $CLUSTER_COMPUTE_NAME
build_id: $CLUSTER_ENV_ID
ray_serve_config:
  import_path: deploy.services.service:entrypoint
  runtime_env:
    working_dir: .
    upload_path: $UPLOAD_PATH
    env_vars:
      EXPERIMENT_NAME: llm
      BUCKET: $BUCKET
      RUN_ID: $RUN_ID
