### Git instructions
```
git clone https://github.com/anyscale/mmwl-playground.git mwml-playground
cd mwml-playground
```

### Virtual environment
```bash
python3 -m venv venv  # use Python >= 3.9
source venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
```

> We highly recommend using Python `3.9.1` (required: `>=3.9`). You can use [pyenv](https://github.com/pyenv/pyenv) (mac) or [pyenv-win](https://github.com/pyenv-win/pyenv-win) (windows) to quickly install and set local python versions for this project.

### Notebook

Execute the commands below inside the virtual environment to load the files necessary for running through our [notebook](notebooks/madewithml.ipynb).

```bash
python3 -m pip install -e .
jupyter nbextension enable --py --sys-prefix widgetsnbextension
```

### Scripts

Execute the commands below inside the virtual environment to load the files necessary for running through our entire application.

```bash
python3 -m pip install -e ".[dev]"
pre-commit install
pre-commit autoupdate
```