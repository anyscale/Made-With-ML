
## Set up

### Git instructions
```
git clone https://github.com/anyscale/mlops-course.git mlops-course
cd mlops-course
```

### Environment
```bash
python3 -m venv venv  # use Python >= 3.9
source venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -e ".[dev]"
```

> We highly recommend using Python `3.9.1` (required: `>=3.9`). You can use [pyenv](https://github.com/pyenv/pyenv) (mac) or [pyenv-win](https://github.com/pyenv-win/pyenv-win) (windows) to quickly install and set local python versions for this project.

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
train_loop_config='{
  "dropout_p": 0.5,
  "lr": 1e-4,
  "lr_factor": 0.8,
  "lr_patience": 3
}'
python src/madewithml/train.py llm \
    $train_loop_config \
    --num-cpu-workers 6 \
    --num-gpu-workers 0 \
    --num-epochs 1 \
    --batch-size 128
```

### Tuning experiment
```bash
initial_params='[{
    "train_loop_config": {
        "dropout_p": 0.5,
        "lr": 1e-4,
        "lr_factor": 0.8,
        "lr_patience": 3
    }
}]'
python src/madewithml/tune.py llm \
    $initial_params \
    --num-runs 2 \
    --num-cpu-workers 6 \
    --num-gpu-workers 0 \
    --num-epochs 1 \
    --batch-size 128
```

### View/compare experiments (MLflow)
```bash
MODEL_REGISTRY=$(python -c "
from config import config
print(config.MODEL_REGISTRY)")
echo "MODEL_REGISTRY: $MODEL_REGISTRY"

mlflow server -h 0.0.0.0 -p 8000 --backend-store-uri $MODEL_REGISTRY
```

### Evaluation
To save the evaluation metrics to a file, just add `> metrics.json` to the end of the command below.
```bash
RUN_ID=$(python -c "
from madewithml import predict
run_id = predict.get_best_run_id(experiment_name='llm', metric='val_loss', direction='ASC')
print(run_id)")
echo "Run ID: $RUN_ID"

python src/madewithml/evaluate.py \
    --run-id $RUN_ID \
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
RUN_ID=$(python -c "
from madewithml import predict
run_id = predict.get_best_run_id(experiment_name='llm', metric='val_loss', direction='ASC')
print(run_id)")
echo "Run ID: $RUN_ID"

# Predict
python src/madewithml/predict.py \
    --title "Transfer learning with transformers" \
    --description "Using transformers for transfer learning on text classification tasks." \
    --run-id $RUN_ID
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

# Get run ID
RUN_ID=$(python -c "
from madewithml import predict
run_id = predict.get_best_run_id(experiment_name='llm', metric='val_loss', direction='ASC')
print(run_id)")
echo "Run ID: $RUN_ID"

# Run application
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
```
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
run_id = predict.get_best_run_id(experiment_name='llm', metric='val_loss', direction='ASC')
print(run_id)")
echo "Run ID: $RUN_ID"
pytest --run-id=$RUN_ID tests/model --disable-warnings
```

## FAQ

### Jupyter notebook kernels

Issues with configuring the notebooks with jupyter? By default, jupyter will use the kernel with our virtual environment but we can also manually add it to jupyter:
```bash
python3 -m ipykernel install --user --name=venv
```
Now we can open up a notebook → Kernel (top menu bar) → Change Kernel → `venv`. To ever delete this kernel, we can do the following:
```bash
jupyter kernelspec list  # show see our venv
jupyter kernelspec uninstall venv
```
