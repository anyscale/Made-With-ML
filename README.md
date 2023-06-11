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

### Cluster

A cluster is a [head node](https://docs.ray.io/en/latest/cluster/key-concepts.html#head-node) (manages the cluster) connected to a set of [worker nodes](https://docs.ray.io/en/latest/cluster/key-concepts.html#head-node) (CPU, GPU, etc.). These clusters can be fixed in size or [autoscale](https://docs.ray.io/en/latest/cluster/key-concepts.html#cluster-autoscaler) up and down based on our application's compute needs.


<details>
  <summary>Local</summary><br>
  Your personal laptop (single machine) will act as the cluster, where one CPU will be the head node and some of the remaining CPU will be the worker nodes. All of the code in this course will work in any personal laptop though it will be slower than executing the same workloads on a larger cluster.
</details>

<details open>
  <summary>Anyscale</summary><br>

  We can create an [Anyscale Workspace](https://docs.anyscale.com/develop/workspaces/get-started) using the [webpage UI](https://console.anyscale.com/o/anyscale-internal/workspaces/add/blank):

  ```md
  - Workspace name: madewithml
  - Project: madewithml
  - Cluster environment name: madewithml-cluster-env
  # Toggle `Select from saved configurations`
  - Compute config: madewithml-cluster-compute
  ```

  Alternatively, we can use the [CLI](https://docs.anyscale.com/reference/anyscale-cli) to create the workspace:

  ```bash
  anyscale workspace create \
    --name madewithml \
    --project-id "prj_hgqw2dlbcn9lt97mulm2rrf8tp" \
    --cloud-id "cld_htln18yuv82lz5he81urmkhgvs" \
    --compute-config-id "cpt_lngp2rax6yaxgegmrlpbvfnp3q" \
    --cluster-env-build-id "bld_7iu2u4azlqyjey9q2rmg52fqgr"
  ```
</details>

<details>
  <summary>Other</summary><br>

  If you don't want to do this course locally or via Anyscale, you have the following options:

  - On [AWS and GCP](https://docs.ray.io/en/latest/cluster/vms/index.html#cloud-vm-index). Community-supported Azure and Aliyun integrations also exist.
  - On [Kubernetes](https://docs.ray.io/en/latest/cluster/kubernetes/index.html#kuberay-index), via the officially supported KubeRay project.
  - Deploy Ray manually [on-prem](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/on-premises.html#on-prem) or onto platforms [not listed here](https://docs.ray.io/en/latest/cluster/vms/user-guides/community/index.html#ref-cluster-setup).

</details>

### Git clone

```bash
git clone https://github.com/anyscale/mlops-course.git .
git checkout -b dev
export PYTHONPATH=$PYTHONPATH:$PWD
```

### Virtual environment

<details>
  <summary>Local</summary><br>

  ```bash
  python3 -m venv venv  # recommend using Python 3.10
  source venv/bin/activate  # on Windows: venv\Scripts\activate
  python3 -m pip install --upgrade pip setuptools wheel
  python3 -m pip install -r requirements.txt
  export PYTHONPATH=$PYTHONPATH:$PWD  # on Windows: set PYTHONPATH=%PYTHONPATH%;C:$PWD
  pre-commit install
  pre-commit autoupdate
  ```

  > Highly recommend using Python `3.10` and using [pyenv](https://github.com/pyenv/pyenv) (mac) or [pyenv-win](https://github.com/pyenv-win/pyenv-win) (windows).

</details>

<details open>
  <summary>Anyscale</summary><br>

  Our environment with the appropriate Python version and libraries is already all set for us through the cluster environment we're using when setting up our Anyscale Workspace.
  ```bash
  export PYTHONPATH=$PYTHONPATH:$PWD  # on Windows: set PYTHONPATH=%PYTHONPATH%;C:$PWD
  pre-commit install
  pre-commit autoupdate
  ```

</details>


## Workloads
1. Start by exploring the interactive [jupyter notebook](notebooks/madewithml.ipynb) to interactively walkthrough the core machine learning workloads.

<details>
  <summary>Local</summary><br>

  ```bash
  # Start notebook
  jupyter lab notebooks/madewithml.ipynb
```

</details>

<details open>
  <summary>Anyscale</summary><br>

  Click on the Jupyter icon at the top right corner of our Anyscale Workspace page and this will open up our JupyterLab instance in a new tab. Then navigate to the `notebooks` directory and open up the `madewithml.ipynb` notebook.

</details>

2. Then execute the same workloads using the clean Python scripts following software engineering best practices (testing, documentation, logging, serving, versioning, etc.)

**Note**: Change the `--num-workers`, `--cpu-per-worker`, and `--gpu-per-worker` input argument values below based on your system's resources. For example, if you're on a local laptop, a reasonable configuration would be `--num-workers 6 --cpu-per-worker 1 --gpu-per-worker 0`.

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

<details>
  <summary>Local</summary><br>

  ```bash
  export MODEL_REGISTRY=$(python -c "from madewithml import config; print(config.MODEL_REGISTRY)")
  mlflow server -h 0.0.0.0 -p 8080 --backend-store-uri $MODEL_REGISTRY
  ```

</details>

<details open>
  <summary>Anyscale</summary><br>

  Work in progress until we get EFS on production clusters w/ port forwarding or we set up an MLflow server ourselves. A temporary workaround is to sync our `$MODEL_REGISTRY` with a S3 bucket and then pull the contents of that bucket locally to view it with the MLflow UI using the same commands above.

</details>

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
  "timestamp": "June 09, 2023 09:26:18 AM",
  "run_id": "6149e3fec8d24f1492d4a4cabd5c06f6",
  "overall": {
    "precision": 0.9076136428670714,
    "recall": 0.9057591623036649,
    "f1": 0.9046792827719773,
    "num_samples": 191.0
  },
...
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

<details>
  <summary>Local</summary><br>

  ```bash
  # Start
  ray start --head
  ```

  ```bash
  # Set up
  export EXPERIMENT_NAME="llm"
  export RUN_ID=$(python madewithml/predict.py get-best-run-id --experiment-name $EXPERIMENT_NAME --metric val_loss --mode ASC)
  python madewithml/serve.py --run_id $RUN_ID
  ```

  While the application is running, we can use it via cURL, Python, etc.:

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

  ```bash
  ray stop  # shutdown
  ```

</details>

<details open>
  <summary>Anyscale</summary><br>

  In Anyscale Workspaces, Ray is already running so we don't have to manually start/shutdown like we have to do locally.

  ```bash
  # Set up
  export EXPERIMENT_NAME="llm"
  export RUN_ID=$(python madewithml/predict.py get-best-run-id --experiment-name $EXPERIMENT_NAME --metric val_loss --mode ASC)
  python madewithml/serve.py --run_id $RUN_ID
  ```

  While the application is running, we can use it via cURL, Python, etc.:

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

</details>

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

From this point onwards, in order to deploy our application into production, we'll need to either be on Anyscale Workspaces or on a cluster on [cloud VMs](https://docs.ray.io/en/latest/cluster/vms/index.html#cloud-vm-index) / [on-prem](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/on-premises.html#on-prem), [etc](https://docs.ray.io/en/latest/cluster/vms/user-guides/community/index.html#ref-cluster-setup). If not on Anyscale, the commands will be [slightly different](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/index.html) but the concepts will be the same.

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
export CLUSTER_ENV_NAME="madewithml-cluster-env"
export CLUSTER_COMPUTE_NAME="madewithml-cluster-compute"
```

2. Create the cluster env and compute (if changed/needed)
```bash
anyscale cluster-env build deploy/cluster_env.yaml --name $CLUSTER_ENV_NAME
anyscale cluster-compute create deploy/cluster_compute.yaml --name $CLUSTER_COMPUTE_NAME
```

3. Anyscale Jobs
```bash
anyscale job submit deploy/jobs/test_code.yaml
anyscale job submit deploy/jobs/test_data.yaml
anyscale job submit deploy/jobs/train_model.yaml
anyscale job submit deploy/jobs/evaluate_model.yaml
anyscale job submit deploy/jobs/test_model.yaml
```

4. Anyscale Services
```bash
# Rollout service
anyscale service rollout -f deploy/services/serve.yaml

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
