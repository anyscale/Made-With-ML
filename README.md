# MLOps Course

Learn how to combine machine learning with software engineering best practices to develop, deploy and maintain ML applications in production.

> MLOps concepts are interweaved and cannot be run in isolation, so be sure to complement the code in this repository with the detailed [MLOps lessons](https://madewithml.com/#mlops).

<div align="left">
    <a target="_blank" href="https://madewithml.com/"><img src="https://img.shields.io/badge/Subscribe-30K-brightgreen"></a>&nbsp;
    <a target="_blank" href="https://github.com/GokuMohandas/Made-With-ML"><img src="https://img.shields.io/github/stars/GokuMohandas/Made-With-ML.svg?style=social&label=Star"></a>&nbsp;
    <a target="_blank" href="https://www.linkedin.com/in/goku"><img src="https://img.shields.io/badge/style--5eba00.svg?label=LinkedIn&logo=linkedin&style=social"></a>&nbsp;
    <a target="_blank" href="https://twitter.com/GokuMohandas"><img src="https://img.shields.io/twitter/follow/GokuMohandas.svg?label=Follow&style=social"></a>
    <br>
</div>

- Lessons: https://madewithml.com/#mlops
- Code: [GokuMohandas/mlops-course](https://github.com/GokuMohandas/mlops-course)

## Set up

### Git clone
```
git clone https://github.com/anyscale/mlops-course.git mlops-course
cd mlops-course
```

### Virtual environment
> Highly recommend using Python `3.10` and using [pyenv](https://github.com/pyenv/pyenv) (mac) or [pyenv-win](https://github.com/pyenv-win/pyenv-win) (windows).
```bash
python3 -m venv venv  # recommend using Python 3.10
source venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -e ".[dev]"
pre-commit install
pre-commit autoupdate
```

### Install Ray
Install Ray from the [latest nightly wheel](https://docs.ray.io/en/latest/ray-overview/installation.html#daily-releases-nightlies) for your specific OS.
```bash
# MacOS (arm64)
python -m pip install -U "ray[air] @ https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp310-cp310-macosx_11_0_arm64.whl"
```

## Workloads
1. Start by exploring the interactive [jupyter notebook](notebooks/madewithml.ipynb) to interactively walkthrough the core machine learning workloads.
```bash
# Start notebook
jupyter lab notebooks/madewithml.ipynb
```
2. Then execute the same workloads using the clean Python scripts following software engineering best practices (testing, documentation, logging, serving, versioning, etc.)

**Note**: Change the `--use-gpu`, `--num-cpu-workers` and `--num-gpu-workers` configurations based on your system's resources.

### Train a single model
```bash
EXPERIMENT_NAME="llm"
DATASET_LOC="https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/madewithml/dataset.csv"
TRAIN_LOOP_CONFIG='{"dropout_p": 0.5, "lr": 1e-4, "lr_factor": 0.8, "lr_patience": 3}'
python src/madewithml/train.py \
    "$EXPERIMENT_NAME" \
    "$DATASET_LOC" \
    "$TRAIN_LOOP_CONFIG" \
    --use-gpu \
    --num-cpu-workers 10 \
    --num-gpu-workers 2 \
    --num-epochs 10 \
    --batch-size 256
```

### Tuning experiment
```bash
EXPERIMENT_NAME="llm"
DATASET_LOC="https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/madewithml/dataset.csv"
TRAIN_LOOP_CONFIG='{"dropout_p": 0.5, "lr": 1e-4, "lr_factor": 0.8, "lr_patience": 3}'
INITIAL_PARAMS="[{\"train_loop_config\": $TRAIN_LOOP_CONFIG}]"
python src/madewithml/tune.py \
    "$EXPERIMENT_NAME" \
    "$DATASET_LOC" \
    "$INITIAL_PARAMS" \
    --num-runs 2 \
    --use-gpu \
    --num-cpu-workers 10 \
    --num-gpu-workers 2 \
    --num-epochs 10 \
    --batch-size 256
```

### Experiment tracking
```bash
MODEL_REGISTRY=$(python -c "from madewithml import config; print(config.MODEL_REGISTRY)")
mlflow server -h 0.0.0.0 -p 8000 --backend-store-uri $MODEL_REGISTRY
```

### Evaluation
```bash
EXPERIMENT_NAME="llm"
RUN_ID=$(python src/madewithml/predict.py get-best-run-id --experiment-name $EXPERIMENT_NAME --metric val_loss --mode ASC)
HOLDOUT_LOC="https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/madewithml/holdout.csv"
python src/madewithml/evaluate.py \
    --run-id $RUN_ID \
    --dataset-loc $HOLDOUT_LOC \
    --num-cpu-workers 2
```
```json
{
  "precision": 0.9164145614485539,
  "recall": 0.9162303664921466,
  "f1": 0.9152388901535271
}
```

### Inference
```bash
# Get run ID
EXPERIMENT_NAME="llm"
RUN_ID=$(python src/madewithml/predict.py get-best-run-id --experiment-name $EXPERIMENT_NAME --metric val_loss --mode ASC)
python src/madewithml/predict.py predict \
    --run-id $RUN_ID \
    --title "Transfer learning with transformers" \
    --description "Using transformers for transfer learning on text classification tasks."
```
```json
[{
  "prediction": [
    "natural-language-processing"
  ],
  "probabilities": {
    "computer-vision": 0.0009767753,
    "mlops": 0.0008223939,
    "natural-language-processing": 0.99762577,
    "other": 0.000575123
  }
}]
```

### Serve
```bash
# Set up
ray start --head  # already running if using Anyscale
EXPERIMENT_NAME="llm"
RUN_ID=$(python src/madewithml/predict.py get-best-run-id --experiment-name $EXPERIMENT_NAME --metric val_loss --mode ASC)
python src/madewithml/serve.py --run_id $RUN_ID
```

While the application is running, we can use it via cURL, Python, etc.
```bash
# via cURL
curl -X POST -H "Content-Type: application/json" -d '{
  "title": "Transfer learning with transformers",
  "description": "Using transformers for transfer learning on text classification tasks."
}' http://127.0.0.1:8000/
```
```python
# via Python
import json
import requests
title = "Transfer learning with transformers"
description = "Using transformers for transfer learning on text classification tasks."
json_data = json.dumps({"title": title, "description": description})
requests.post("http://127.0.0.1:8000/", data=json_data).json()
```

Once we're done, we can shut down the application:
```bash
# Shutdown
ray stop
```

### Testing
```bash
# Code
python3 -m pytest tests/code --verbose --disable-warnings

# Data
DATASET_LOC="https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/madewithml/dataset.csv"
pytest --dataset-loc=$DATASET_LOC tests/data --verbose --disable-warnings

# Model
EXPERIMENT_NAME="llm"
RUN_ID=$(python src/madewithml/predict.py get-best-run-id --experiment-name $EXPERIMENT_NAME --metric val_loss --mode ASC)
pytest --run-id=$RUN_ID tests/model --verbose --disable-warnings
```

## Workspaces
```bash
# Instructions inside Workspaces
git clone https://github.com/anyscale/mlops-course.git .
pip install -e ".[dev]"
pip install -U "ray[air] @ https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp310-cp310-manylinux2014_x86_64.whl"
```

## Deploy

### Authentication
``` bash
export ANYSCALE_HOST=https://console.anyscale-staging.com
export ANYSCALE_CLI_TOKEN=$YOUR_CLI_TOKEN  # retrieved from Anyscale credentials page
export AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID  # retrieved from AWS IAM
export AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
export AWS_SESSION_TOKEN=$AWS_SESSION_TOKEN
```

### Workloads

1. Set env vars
```bash
export PROJECT_NAME="madewithml"
export CLUSTER_ENV_NAME="madewithml-cluster-env"
export S3_BUCKET="s3://madewithml"
```

2. Create the project
```bash
anyscale project create -n $PROJECT_NAME
```

3. Replace vars in configs
```bash
# Replace vars in configs (jobs/*.yaml and services/*.yaml)
python deploy/utils.py get-project-id --project-name $PROJECT_NAME
python deploy/utils.py get-latest-cluster-env-build-id --cluster-env-name $CLUSTER_ENV_NAME
```

4. Test code
```bash
# Manual
anyscale job submit deploy/jobs/test_code.yaml

# Dynamic
python deploy/utils.py submit-job \
  --yaml-config-fp deploy/jobs/test_code.yaml \
  --cluster-env-name $CLUSTER_ENV_NAME
```

5. Test data
```bash
# Manual
anyscale job submit deploy/jobs/test_data.yaml

# Dynamic
python deploy/utils.py submit-job \
  --yaml-config-fp deploy/jobs/test_data.yaml \
  --cluster-env-name $CLUSTER_ENV_NAME
```

6. Train model
```bash
# Manual
anyscale job submit deploy/jobs/train.yaml

# Dynamic
python deploy/utils.py submit-job \
  --yaml-config-fp deploy/jobs/train.yaml \
  --cluster-env-name $CLUSTER_ENV_NAME
```

7. Evaluate model
```bash
# Manual
anyscale job submit deploy/jobs/evaluate.yaml

# Dynamic
python deploy/utils.py submit-job \
  --yaml-config-fp deploy/jobs/evaluate.yaml \
  --cluster-env-name $CLUSTER_ENV_NAME
```

8. Test model
```bash
```

9. Compare to prod
```bash
```

10. Deploy service
```bash
```
------------------------------------------------------------------------------------------------------------------------

### Setup
```bash
export PROJECT_NAME="mlops-course"  # project name should match with repository name
anyscale project create -n $PROJECT_NAME
export PROJECT_ID=$(python deploy/utils/utils.py get-project-id --project-name $PROJECT_NAME)
export CLUSTER_ENV_NAME="madewithml-cluster-env"
export CLUSTER_ENV_BUILD_ID=$(python deploy/utils/utils.py get-latest-cluster-env-build-id --cluster-env-name $CLUSTER_ENV_NAME)
export S3_BUCKET="s3://goku-mlops"
export UUID=$(python -c 'import uuid; print(str(uuid.uuid4())[:8])')
anyscale cluster-env build deploy/cluster_env.yaml --name madewithml-cluster-env
```


### Evaluation job
Either `experiment_name` or `run_id` must be provided. If both are provided, `run_id` will be used. And if only `experiment_id` is provided, the best run from the experiment will be used.

```bash
python deploy/utils/job_submit.py deploy/jobs/evaluate.yaml \
  uuid=$UUID \
  project_id=$PROJECT_ID \
  build_id=$CLUSTER_ENV_BUILD_ID \
  upload_path=$S3_BUCKET/workingdir/job \
  s3_bucket=$S3_BUCKET \
  experiment_name=llm
```


### Services

```bash
# Set up
export SERVICE_CONFIG="deploy/services/service.yaml"
export SERVICE_NAME="madewithml-service"

# Rollout
anyscale service rollout -f $SERVICE_CONFIG --name $SERVICE_NAME

# Query (retrieved from Service endpoint generated from command above)
curl -X POST -H "Content-Type: application/json" -H "Authorization: Bearer $SECRET_TOKEN" -d '{
  "title": "Transfer learning with transformers",
  "description": "Using transformers for transfer learning on text classification tasks."
}' $SERVICE_ENDPOINT

# Rollback (to previous version of the Service)
anyscale service rollback -f $SERVICE_CONFIG --name $SERVICE_NAME

# Terminate
anyscale service terminate --name $SERVICE_NAME
```







## CI/CD

We're not going to manually deploy our application every time we make a change. Instead, we'll automate this process using GitHub Actions!

1. We'll start by adding the necessary credentials to the [`/settings/secrets/actions`](https://github.com/GokuMohandas/mlops-course/settings/secrets/actions) page of our GitHub repository.

``` bash
export ANYSCALE_HOST=https://console.anyscale-staging.com
export ANYSCALE_CLI_TOKEN=$YOUR_CLI_TOKEN  # retrieved from https://console.anyscale-staging.com/o/anyscale-internal/credentials
export AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID  # retrieved from AWS IAM
export AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
export AWS_SESSION_TOKEN=$AWS_SESSION_TOKEN
```

2. Next, we'll create the different GitHub action workflows:
- Test code, data, model: [`/.github/workflows/testing.yaml`](/.github/workflows/testing.yaml)
- Train model: [`/.github/workflows/training.yaml`](/.github/workflows/training.yaml)
- Evaluate model: [`/.github/workflows/evaluation.yaml`](/.github/workflows/evaluation.yaml)
- Deploy service: [`/.github/workflows/serving.yaml`](/.github/workflows/serving.yaml)

3. Our different workflows will be triggered when we make a change to our repository and push to a branch and trigger a PR to our main branch. Inside that PR, we'll see the different workflows' results, which we can use to decide if we should deploy our new model or not. Merging a PR to main will update the current deployed model with our new model.
> We can also manually trigger them from the [`/actions`](https://github.com/GokuMohandas/mlops-course/actions) tab of our GitHub repository

Our Git workflow looks like this:
```bash
# Checkout new branch
git checkout -b $BRANCH_NAME  # ex. dev

# Develop on branch then push to remote
git add .
git commit -m "message"
git push origin $BRANCH_NAME

# Create and merge PR to main branch (on github.com)
https://github.com/$USERNAME/$REPO/pull/new/$BRANCH_NAME

# Merge PR to main branch (on local)
git checkout main
git pull origin main

# Update branch
git checkout $BRANCH_NAME
git merge main  # ready to develop again
```


## FAQ

### Jupyter notebook kernels
Issues with configuring the notebooks with jupyter? By default, jupyter will use the kernel with our virtual environment but we can also manually add it to jupyter:
```bash
python3 -m ipykernel install --user --name=venv
```
Now we can open up a notebook → Kernel (top menu bar) → Change Kernel → `venv`. To ever delete this kernel, we can do the following:
```bash
jupyter kernelspec list
jupyter kernelspec uninstall venv
```
