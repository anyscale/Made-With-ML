
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
jupyter lab notebooks/finetune-llm.ipynb
```

### Scripts

Execute the commands below inside the virtual environment to load the files necessary for running through our entire application.

```bash
python3 -m pip install -e ".[dev]"
pre-commit install
pre-commit autoupdate
```

## Workloads

### Train a single model
```python
from madewithml import train
result = train.train_model(experiment_name="llm", num_workers=9, use_gpu=False)
```

### Tune multiple models
```python
from madewithml import tune
result_grid = tune.tune_models(experiment_name="llm", num_runs=10, num_workers=9, use_gpu=False)
```

### Inference
```python
from madewithml import predict
predict.predict(texts=["Transfer learning with transformers for text classification."])
```

## Dashboards

### Ray
```python
import ray
ray.init()
```
Go to [http://127.0.0.1:8265](http://127.0.0.1:8265)

### MLflow
```bash
mlflow server -h 0.0.0.0 -p 8000 --backend-store-uri $PWD/mlruns
```
Go to [http://0.0.0.0:8000](http://0.0.0.0:8000)

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

