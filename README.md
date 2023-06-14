<div align="center">
<h1><img width="30" src="https://madewithml.com/static/images/rounded_logo.png">&nbsp;<a href="https://madewithml.com/">Made With ML</a></h1>
Design · Develop · Deploy · Iterate
<br>
Join 40K+ developers in learning how to responsibly <a href="https://madewithml.com/about/">deliver value</a> with ML.
    <br>
</div>

<br>

<div align="center">
    <a target="_blank" href="https://madewithml.com/"><img src="https://img.shields.io/badge/Subscribe-40K-brightgreen"></a>&nbsp;
    <a target="_blank" href="https://github.com/GokuMohandas/Made-With-ML"><img src="https://img.shields.io/github/stars/GokuMohandas/Made-With-ML.svg?style=social&label=Star"></a>&nbsp;
    <a target="_blank" href="https://www.linkedin.com/in/goku"><img src="https://img.shields.io/badge/style--5eba00.svg?label=LinkedIn&logo=linkedin&style=social"></a>&nbsp;
    <a target="_blank" href="https://twitter.com/GokuMohandas"><img src="https://img.shields.io/twitter/follow/GokuMohandas.svg?label=Follow&style=social"></a>
    <br>
    🔥&nbsp; Among the <a href="https://github.com/GokuMohandas/Made-With-ML" target="_blank">top MLOps</a> repositories on GitHub
</div>

<br>
<hr>

- Lessons: https://madewithml.com/
- Code: [GokuMohandas/Made-With-ML](https://github.com/GokuMohandas/Made-With-ML)

## Overview

In this course, we'll journey from experimentation (model design + development) towards production (model deployment + iteration) so that we can deliver value with machine learning. We'll do this iteratively by motivating the components that will enable us to build a *reliable* production system.

Along the way, we'll address some of the largest obstacles that *used to* prevent ML from being easily and reliably deployed in production. And how [Ray](https://ray.io/), an open-source framework to scale AI applications that's used by ML teams at companies like [Spotify](https://engineering.atspotify.com/2023/02/unleashing-ml-innovation-at-spotify-with-ray/), [OpenAI](https://thenewstack.io/how-ray-a-distributed-ai-framework-helps-power-chatgpt/), [Instacart](https://tech.instacart.com/distributed-machine-learning-at-instacart-4b11d7569423), etc., makes it easy to overcome these obstacles.

<blockquote>
  <img width=20 src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/YouTube_full-color_icon_%282017%29.svg/640px-YouTube_full-color_icon_%282017%29.svg.png">&nbsp; Be sure to watch the video below for an in-depth overview of what we'll be building.
</blockquote>

<div align="center">
  <a href="https://www.youtube.com/watch?v=7XF-h-oUR2Q"><img src="https://img.youtube.com/vi/7XF-h-oUR2Q/0.jpg" alt="Course overview video"></a>
</div>

## Set up

### Cluster

A cluster is a [head node](https://docs.ray.io/en/latest/cluster/key-concepts.html#head-node) (manages the cluster) connected to a set of [worker nodes](https://docs.ray.io/en/latest/cluster/key-concepts.html#head-node) (CPU, GPU, etc.). These clusters can be fixed in size or [autoscale](https://docs.ray.io/en/latest/cluster/key-concepts.html#cluster-autoscaler) up and down based on our application's compute needs.

> **Note**: We will have instructions for both local laptop and Anyscale clusters for the sections below, so be sure to toggle the ► dropdown based on what you're using. (Anyscale instructions will be toggled on by default).

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
  <summary>Other (cloud platforms, K8s, on-prem)</summary><br>

  If you don't want to do this course locally or via Anyscale, you have the following options:

  - On [AWS and GCP](https://docs.ray.io/en/latest/cluster/vms/index.html#cloud-vm-index). Community-supported Azure and Aliyun integrations also exist.
  - On [Kubernetes](https://docs.ray.io/en/latest/cluster/kubernetes/index.html#kuberay-index), via the officially supported KubeRay project.
  - Deploy Ray manually [on-prem](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/on-premises.html#on-prem) or onto platforms [not listed here](https://docs.ray.io/en/latest/cluster/vms/user-guides/community/index.html#ref-cluster-setup).

</details>

### Git clone

```bash
git clone https://github.com/anyscale/Made-With-ML.git .
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

### Install Ray
Install Ray from the [latest nightly wheel](https://docs.ray.io/en/latest/ray-overview/installation.html#daily-releases-nightlies) for your specific OS.
```bash
# MacOS (arm64)
python -m pip install -U "ray[air] @ https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp310-cp310-macosx_11_0_arm64.whl"
```


## Notebook

Start by exploring the interactive [jupyter notebook](notebooks/madewithml.ipynb) to interactively walkthrough the core machine learning workloads.

<div align="center">
  <img src="https://madewithml.com/static/images/mlops/experimentation.png">
</div>

<details>
  <summary>Local</summary><br>

  ```bash
  # Start notebook
  jupyter lab notebooks/madewithml.ipynb
```

</details>

<details open>
  <summary>Anyscale</summary><br>

  Click on the Jupyter icon &nbsp;<img width=15 src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Jupyter_logo.svg/1200px-Jupyter_logo.svg.png">&nbsp; at the top right corner of our Anyscale Workspace page and this will open up our JupyterLab instance in a new tab. Then navigate to the `notebooks` directory and open up the `madewithml.ipynb` notebook.

</details>


## Scripts

Now we'll execute the same workloads using the clean Python scripts following software engineering best practices (testing, documentation, logging, serving, versioning, etc.) The code we've implemented in our notebook will be refactored into the following scripts:

```bash
madewithml
├── config.py
├── data.py
├── evaluate.py
├── models.py
├── predict.py
├── serve.py
├── train.py
├── tune.py
└── utils.py
```

**Note**: Change the `--num-workers`, `--cpu-per-worker`, and `--gpu-per-worker` input argument values below based on your system's resources. For example, if you're on a local laptop, a reasonable configuration would be `--num-workers 6 --cpu-per-worker 1 --gpu-per-worker 0`.

### Training
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

### Tuning
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

### Serving

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

## Deployment

From this point onwards, in order to deploy our application into production, we'll need to either be on Anyscale or on a [cloud VMs](https://docs.ray.io/en/latest/cluster/vms/index.html#cloud-vm-index) / [on-prem](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/on-premises.html#on-prem) cluster you manage yourself. If not on Anyscale, the commands will be [slightly different](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/index.html) but the concepts will be the same.

<div align="center">
  <img src="https://madewithml.com/static/images/mlops/manual.png">
</div>

### Authentication
> We **do not** need to set these credentials if we're using Anyscale Workspaces but we do if we're running this locally or on a cluster outside of where our Anyscale Jobs and Services are configured to run (via `cluster_env`).
``` bash
export ANYSCALE_HOST=https://console.anyscale-staging.com
export ANYSCALE_CLI_TOKEN=$YOUR_CLI_TOKEN  # retrieved from Anyscale credentials page
export AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID  # retrieved from AWS IAM
export AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
export AWS_SESSION_TOKEN=$AWS_SESSION_TOKEN
```

### Anyscale Jobs

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

3. Job submissions
```bash
anyscale job submit deploy/jobs/test_code.yaml
anyscale job submit deploy/jobs/test_data.yaml
anyscale job submit deploy/jobs/train_model.yaml
anyscale job submit deploy/jobs/evaluate_model.yaml
anyscale job submit deploy/jobs/test_model.yaml
```

### Anyscale Services

```bash
# Rollout service
anyscale service rollout -f deploy/services/serve_model.yaml

# Query
curl -X POST -H "Content-Type: application/json" -H "Authorization: Bearer $SECRET_TOKEN" -d '{
  "title": "Transfer learning with transformers",
  "description": "Using transformers for transfer learning on text classification tasks."
}' $SERVICE_ENDPOINT

# Rollback (to previous version of the Service)
anyscale service rollback -f $SERVICE_CONFIG --name $SERVICE_NAME

# Terminate
anyscale service terminate --name $SERVICE_NAME
```

### CI/CD

We're not going to manually deploy our application every time we make a change. Instead, we'll automate this process using GitHub Actions!

<div align="center">
  <img src="https://madewithml.com/static/images/mlops/cicd.png">
</div>

1. We'll start by adding the necessary credentials to the [`/settings/secrets/actions`](https://github.com/GokuMohandas/Made-With-ML/settings/secrets/actions) page of our GitHub repository.

``` bash
export ANYSCALE_HOST=https://console.anyscale.com
export ANYSCALE_CLI_TOKEN=$YOUR_CLI_TOKEN  # retrieved from https://console.anyscale.com/o/anyscale-internal/credentials
export AWS_REGION=us-west-2
export IAM_ROLE=arn:aws:iam::959243851260:role/github-action-madewithml
```

2. Now we can make changes to our code (not on `main` branch) and push them to GitHub. When we start a PR from this branch to our `main` branch, this will trigger the [workloads workflow](/.github/workflows/workloads.yaml). If the workflow goes well, this will produce comments with the training, evaluation and current prod evaluation (if applicable) directly on the PR.

3. After we compare our new experiment with what is currently in prod (if applicable), we can merge the PR into the `main` branch. This will trigger the [deployment workflow](/.github/workflows/deployment.yaml) which will rollout our new service to production!

### Continual learning

With our CI/CD workflow in place to deploy our application, we can now focus on continually improving our model. It becomes really easy to extend on this foundation to connect to scheduled runs (cron), [data pipelines](https://madewithml.com/courses/mlops/data-stack/), [orchestrate workflows](https://madewithml.com/courses/mlops/orchestration/), drift detected through [monitoring](https://madewithml.com/courses/mlops/monitoring/), [online evaluation](https://madewithml.com/courses/mlops/evaluation/#online-evaluation), etc.

<div align="center">
  <img src="https://madewithml.com/static/images/mlops/continual.png">
</div>

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
