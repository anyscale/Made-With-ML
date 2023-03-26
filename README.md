
Using Python `3.9.16` but should work with any version of Python `>=3.7`. I'm running this locally (M1 with 10 CPUs) on a VSCode Jupyter notebook.

### Git instructions
```
git clone https://github.com/anyscale/mmwl-playground.git mwml-playground
cd mwml-playground
```

### Set up
```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
jupyter nbextension enable --py --sys-prefix widgetsnbextension
```

Now you're ready to open the notebook and execute the code.