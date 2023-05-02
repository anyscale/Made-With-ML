
## Set up

### Git instructions
```
git clone https://github.com/anyscale/mmwl-playground.git mwml-playground
cd mwml-playground
```

### Virtual environment
```bash
mkdir src
python3 -m venv venv  # use Python >= 3.9
source venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
```

> We highly recommend using Python `3.9.1` (required: `>=3.9`). You can use [pyenv](https://github.com/pyenv/pyenv) (mac) or [pyenv-win](https://github.com/pyenv-win/pyenv-win) (windows) to quickly install and set local python versions for this project.

### Notebook

Execute the commands below inside the virtual environment to load the files necessary for running through our [notebook](notebooks/madewithml.ipynb).

```bash
python3 -m pip install -e ".[notebook]"
jupyter lab notebooks/finetune_llm.ipynb
```

### Development

Execute the commands below inside the virtual environment to prepare for development.

```bash
python3 -m pip install -e ".[dev]"
pre-commit install
pre-commit autoupdate
```

## Workloads

**Note**: Change the `--use-gpu`, `--num-cpu-workers` and `--num-gpu-workers` configurations based on your system's resources.

### Train a single model
```bash
python src/madewithml/train.py llm \
    --use-gpu \
    --num-cpu-workers 40 \
    --num-gpu-workers 2
```

### Tuning experiment
```bash
python src/madewithml/tune.py llm \
    --num-runs 10 \
    --num-cpu-workers 40 \
    --num-gpu-workers 2
```

### Evaluation
```bash
```

### Inference
```bash
# Get run ID
run_id=$(python -c "
from madewithml import predict
run_id = predict.get_best_run_id(experiment_name='llm', metric='val_loss', direction='ASC')
print(run_id)")
echo "Run ID: $run_id"

# Predict
python src/madewithml/predict.py \
    --title "Transfer learning with transformers" \
    --description "Using transformers for transfer learning on text classification tasks." \
    --run-id $run_id
```
```json
[{
  "pred": [
    "natural-language-processing"
  ],
  "prob": {
    "computer-vision": 0.0009767753,
    "mlops": 0.0008223939,
    "natural-language-processing": 0.99762577,
    "other": 0.000575123
  }
}]
```

## Batch inference (offline)
```

```

## Online inference (Serve)
```bash
# Set up
ray start --head  # already running if using Anyscale

# Get run ID
run_id=$(python -c "
from madewithml import predict
run_id = predict.get_best_run_id(experiment_name='llm', metric='val_loss', direction='ASC')
print(run_id)")
echo "Run ID: $run_id"

# Run application
python src/madewithml/serve.py --run_id $run_id

# Prediction
curl -G \
  --data-urlencode 'title=Transfer learning with transformers for text classification.' \
  --data-urlencode 'description=Using transformers for transfer learning on text classification tasks.' \
  http://127.0.0.1:8000/

# Shutdown
ray stop
```

While the application is running, we can use it via Python as well:
```python
# via Python
import requests
title = "Transfer learning with transformers for text classification."
description = "Using transformers for transfer learning on text classification tasks."
requests.get("http://127.0.0.1:8000/", params={"title": title, "description": description}).json()
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
