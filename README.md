
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
EXPERIMENT_NAME="madewithml-$(openssl rand -hex 4)"
DATASET_LOC="https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/madewithml/dataset.csv"
TRAIN_LOOP_CONFIG='{"dropout_p": 0.5, "lr": 1e-4, "lr_factor": 0.8, "lr_patience": 3}'
echo "Experiment name: $EXPERIMENT_NAME"
python src/madewithml/train.py \
    "$EXPERIMENT_NAME" \
    "$DATASET_LOC" \
    "$TRAIN_LOOP_CONFIG" \
    --use-gpu \
    --num-cpu-workers 40 \
    --num-gpu-workers 2 \
    --num-epochs 10 \
    --batch-size 256
```

### Tuning experiment
```bash
EXPERIMENT_NAME="madewithml-$(openssl rand -hex 4)"
DATASET_LOC="https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/madewithml/dataset.csv"
TRAIN_LOOP_CONFIG='{"dropout_p": 0.5, "lr": 1e-4, "lr_factor": 0.8, "lr_patience": 3}'
INITIAL_PARAMS="[{\"train_loop_config\": $TRAIN_LOOP_CONFIG}]"
echo "Experiment name: $EXPERIMENT_NAME"
python src/madewithml/tune.py \
    "$EXPERIMENT_NAME" \
    "$DATASET_LOC" \
    "$INITIAL_PARAMS" \
    --num-runs 2 \
    --use-gpu \
    --num-cpu-workers 40 \
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
RUN_ID=$(python -c "from madewithml.predict import get_best_run_id as g; print(g('$EXPERIMENT_NAME', 'val_loss', 'ASC'))")
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
RUN_ID=$(python -c "from madewithml.predict import get_best_run_id as g; print(g('$EXPERIMENT_NAME', 'val_loss', 'ASC'))")
python src/madewithml/predict.py \
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
RUN_ID=$(python -c "from madewithml.predict import get_best_run_id as g; print(g('$EXPERIMENT_NAME', 'val_loss', 'ASC'))")
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
python3 -m pytest tests/code --cov src/madewithml --cov-config=pyproject.toml --cov-report html --disable-warnings
make clean
open htmlcov/index.html

# Data
pytest tests/data --disable-warnings

# Model
RUN_ID=$(python -c "
from madewithml import predict
run_id = predict.get_best_run_id(experiment_name='$EXPERIMENT_NAME', metric='val_loss', direction='ASC')
print(run_id)")
echo "Run ID: $RUN_ID"
pytest --run-id=$RUN_ID tests/model --disable-warnings
```

## Production

### Authentication
``` bash
export ANYSCALE_HOST=https://console.anyscale.com
export ANYSCALE_CLI_TOKEN=$YOUR_CLI_TOKEN  # retrieved from https://console.anyscale.com/credentials
```

### Setup
```bash
mkdir deploy
anyscale project create -n $PROJECT_NAME
anyscale cluster-env get --id $CLUSTER_ENV_BUILD_ID > deploy/cluster_env.yml
anyscale compute-config get --id $CLUSTER_COMPUTE_ID > deploy/compute_config.yml

```


Above Set up isn't really relevant for our user journey.
1. We create the cluster environment config and compute config using the UI.
2. We

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
