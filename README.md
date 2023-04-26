
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

### Scripts

Execute the commands below inside the virtual environment to load the files necessary for running through our entire application.

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

### Inference
```bash
python src/madewithml/predict.py \
    --title "Transfer learning with transformers" \
    --description "Using transformers for transfer learning on text classification tasks." \
    --experiment-name llm
```

### Smoke tests
```bash
python src/madewithml/tune.py test \
    --use-gpu \
    --num-cpu-workers 40 \
    --num-gpu-workers 2 \
    --num-runs 1 \
    --num-samples 100 \
    --num-epochs 1 \
    --batch-size 32 \
    --smoke-test
```

## Serve
```bash
# via Bash
ray start --head  # already running if using Anyscale
python src/app/api.py
curl -G \
  --data-urlencode 'title=Transfer learning with transformers for text classification.' \
  --data-urlencode 'description=Using transformers for transfer learning on text classification tasks.' \
  http://127.0.0.1:8000/
ray stop
```
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
