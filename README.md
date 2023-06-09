# Made With ML

Learn how to combine machine learning with software engineering best practices to develop, deploy and maintain ML applications in production.

> MLOps concepts are interweaved and so should not be run in isolation, so be sure to complement the code in this repository with the detailed [MLOps lessons](https://madewithml.com/#mlops).

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
source venv/bin/activate  # on Windows: venv\Scripts\activate
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$PWD  # on Windows: set PYTHONPATH=%PYTHONPATH%;C:$PWD
pre-commit install
pre-commit autoupdate
```

## Workloads
1. Start by exploring the interactive [jupyter notebook](notebooks/madewithml.ipynb) to interactively walkthrough the core machine learning workloads.
```bash
# Start notebook
jupyter lab notebooks/madewithml.ipynb
```
2. Then execute the same workloads using the clean Python scripts following software engineering best practices (testing, documentation, logging, serving, versioning, etc.)

**Note**: Change the `--use-gpu`, `--num-workers`, `cpu_per_worker`, and `--gpu-per-worker` configurations based on your system's resources.

### Train a single model
```bash
export EXPERIMENT_NAME="llm"
export DATASET_LOC="https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/madewithml/dataset.csv"
export TRAIN_LOOP_CONFIG='{"dropout_p": 0.5, "lr": 1e-4, "lr_factor": 0.8, "lr_patience": 3}'
python madewithml/train.py \
    --experiment-name "$EXPERIMENT_NAME" \
    --dataset-loc "$DATASET_LOC" \
    --num-repartitions 3 \
    --train-loop-config "$TRAIN_LOOP_CONFIG" \
    --num-workers 2 \
    --cpu-per-worker 10 \
    --gpu-per-worker 1 \
    --num-epochs 10 \
    --batch-size 256 \
    --results-fp results/training_results.json
```

### Tuning experiment
```bash
export EXPERIMENT_NAME="llm"
export DATASET_LOC="https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/madewithml/dataset.csv"
export TRAIN_LOOP_CONFIG='{"dropout_p": 0.5, "lr": 1e-4, "lr_factor": 0.8, "lr_patience": 3}'
export INITIAL_PARAMS="[{\"train_loop_config\": $TRAIN_LOOP_CONFIG}]"
python madewithml/tune.py \
    --experiment-name "$EXPERIMENT_NAME" \
    --dataset-loc "$DATASET_LOC" \
    --num-repartitions 3 \
    --initial-params "$INITIAL_PARAMS" \
    --num-runs 2 \
    --num-workers 2 \
    --cpu-per-worker 10 \
    --gpu-per-worker 1 \
    --num-epochs 10 \
    --batch-size 256 \
    --results-fp results/tuning_results.json
```

### Experiment tracking
If you've been following the course through Anyscale, be sure to sync the mlflow artifacts from S3 before running the commands below on your local machine. If you've been running too many experiment, you can make the s3 bucket location point to a specific experiment's folder.

> In an actual data science team, you would have a centralized MLFlow server that everyone would use to track their experiments --- which we could easily achieve with  a database backend on top of our S3 artifact root.
```bash
# Sync artifacts from S3 (run on local machine if using Anyscale)
export GITHUB_USERNAME=GokuMohandas
aws s3 sync s3://madewithml/$GITHUB_USERNAME/mlflow $MODEL_REGISTRY
```
```bash
export MODEL_REGISTRY=$(python -c "from madewithml import config; print(config.MODEL_REGISTRY)")
mlflow server -h 0.0.0.0 -p 8000 --backend-store-uri $MODEL_REGISTRY
```

### Evaluation
```bash
export EXPERIMENT_NAME="llm"
export RUN_ID=$(python madewithml/predict.py get-best-run-id --experiment-name $EXPERIMENT_NAME --metric val_loss --mode ASC)
export HOLDOUT_LOC="https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/madewithml/holdout.csv"
python madewithml/evaluate.py \
    --run-id $RUN_ID \
    --dataset-loc $HOLDOUT_LOC \
    --num-repartitions 3 \
    --results-fp results/evaluation_results.json
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
export EXPERIMENT_NAME="llm"
export RUN_ID=$(python madewithml/predict.py get-best-run-id --experiment-name $EXPERIMENT_NAME --metric val_loss --mode ASC)
python madewithml/predict.py predict \
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
export EXPERIMENT_NAME="llm"
export RUN_ID=$(python madewithml/predict.py get-best-run-id --experiment-name $EXPERIMENT_NAME --metric val_loss --mode ASC)
python madewithml/serve.py --run_id $RUN_ID
```

While the application is running, we can use it via cURL, Python, etc.
```bash
# via cURL
curl -X POST -H "Content-Type: application/json" -d '{
  "title": "Transfer learning with transformers",
  "description": "Using transformers for transfer learning on text classification tasks."
}' http://127.0.0.1:8000/predict
```
```python
# via Python
import json
import requests
title = "Transfer learning with transformers"
description = "Using transformers for transfer learning on text classification tasks."
json_data = json.dumps({"title": title, "description": description})
requests.post("http://127.0.0.1:8000/predict", data=json_data).json()
```

> If you're running in a cluster environment where ray is not already running, we'll need to start it up and shut it down:
```bash
ray start --head  # already running if using Anyscale
# Serve operations above
ray stop  # showtdown
```

### Testing
```bash
# Code
python3 -m pytest tests/code --verbose --disable-warnings

# Data
export DATASET_LOC="https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/madewithml/dataset.csv"
pytest --dataset-loc=$DATASET_LOC tests/data --verbose --disable-warnings

# Model
export EXPERIMENT_NAME="llm"
export RUN_ID=$(python madewithml/predict.py get-best-run-id --experiment-name $EXPERIMENT_NAME --metric val_loss --mode ASC)
pytest --run-id=$RUN_ID tests/model --verbose --disable-warnings
```


## Deploy

### Authentication
> We **do not** need to set these credentials if we're using Anyscale Workspaces :)
``` bash
export ANYSCALE_HOST=https://console.anyscale-staging.com
export ANYSCALE_CLI_TOKEN=$YOUR_CLI_TOKEN  # retrieved from Anyscale credentials page
export AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID  # retrieved from AWS IAM
export AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
export AWS_SESSION_TOKEN=$AWS_SESSION_TOKEN
```

### Workloads

1. Set environment variables
```bash
export PROJECT_NAME="madewithml"
export CLUSTER_ENV_NAME="madewithml-cluster-env"
```

2. Create the project and cluster env
```bash
anyscale project create -n $PROJECT_NAME
anyscale cluster-env build deploy/cluster_env.yaml --name $CLUSTER_ENV_NAME
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

# Dynamic (uses latest cluster env build)
python deploy/utils.py submit-job \
  --yaml-config-fp deploy/jobs/test_code.yaml \
  --cluster-env-name $CLUSTER_ENV_NAME
```

5. Test data
```bash
# Manual
anyscale job submit deploy/jobs/test_data.yaml

# Dynamic (uses latest cluster env build)
python deploy/utils.py submit-job \
  --yaml-config-fp deploy/jobs/test_data.yaml \
  --cluster-env-name $CLUSTER_ENV_NAME
```

6. Train model
```bash
# Manual
anyscale job submit deploy/jobs/train_model.yaml

# Dynamic (uses latest cluster env build)
python deploy/utils.py submit-job \
  --yaml-config-fp deploy/jobs/train_model.yaml \
  --cluster-env-name $CLUSTER_ENV_NAME
```

7. Evaluate model
```bash
# Manual
anyscale job submit deploy/jobs/evaluate_model.yaml

# Dynamic (uses latest cluster env build)
python deploy/utils.py submit-job \
  --yaml-config-fp deploy/jobs/evaluate_model.yaml \
  --cluster-env-name $CLUSTER_ENV_NAME
```

8. Test model
```bash
# Manual
anyscale job submit deploy/jobs/test_data.yaml

# Dynamic (uses latest cluster env build)
python deploy/utils.py submit-job \
  --yaml-config-fp deploy/jobs/test_model.yaml \
  --cluster-env-name $CLUSTER_ENV_NAME
```

9. Serve model
```bash
# Manual rollout
anyscale service rollout -f deploy/services/serve.yaml

# Dynamic rollout (uses latest cluster env build)
python deploy/utils.py rollout-service \
  --yaml-config-fp deploy/services/serve.yaml \
  --cluster-env-name $CLUSTER_ENV_NAME

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
export AWS_REGION=us-west-2
export IAM_ROLE=arn:aws:iam::959243851260:role/github-action-madewithml
```

2. Now we can make changes to our code (not on `main` branch) and push them to GitHub. When we start a PR from this branch to our `main` branch, this will trigger the [workloads workflow](/.github/workflows/workloads.yaml). If the workflow goes well, this will produce comments with the training, evaluation and current prod evaluation (if applicable) directly on the PR.

3. After we compare our new experiment with what is currently in prod (if applicable), we can merge the PR into the `main` branch. This will trigger the [deployment workflow](/.github/workflows/deployment.yaml) which will rollout our new service to production!

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
