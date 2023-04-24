# madewithml/utils.py
import os
import json
import mlflow
import numpy as np
from pathlib import Path
import random
import torch
from typing import Any, Dict, List
from urllib.parse import urlparse

import ray
from ray.air import Checkpoint


def set_seeds(seed: int = 42):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    eval("setattr(torch.backends.cudnn, 'deterministic', True)")
    eval("setattr(torch.backends.cudnn, 'benchmark', False)")
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_dict(path: str) -> Dict:
    """Load a dictionary from a JSON's filepath.

    Args:
        path (str): location of file.

    Returns:
        Dict: loaded JSON data.
    """
    with open(path) as fp:
        d = json.load(fp)
    return d


def save_dict(d: Dict, path: str, cls: Any = None, sortkeys: bool = False) -> None:
    """Save a dictionary to a specific location.

    Args:
        d (Dict): data to save.
        path (str): location of where to save the data.
        cls (optional): encoder to use on dict data. Defaults to None.
        sortkeys (bool, optional): whether to sort keys alphabetically. Defaults to False.
    """
    with open(path, "w") as fp:
        json.dump(d, indent=2, fp=fp, cls=cls, sort_keys=sortkeys)
        fp.write("\n")


def get_values(ds: ray.data.Dataset, col: str) -> List:
    """Return a list of values from a specific column in a Ray Dataset.

    Args:
        ds (ray.data.Dataset): Ray Dataset.
        col (str): name of the column to extract values from.

    Returns:
        List: a list of the column's values.
    """
    return ds.select_columns([col]).to_pandas()[col].tolist()


def get_best_checkpoint(run_id: str) ->  ray.train.Checkpoint:
    """Get the best checkpoint (by performance) from a specific run.

    Args:
        run_id (str): ID of the run to get the best checkpoint from.

    Returns:
        ray.train.Checkpoint: Best Checkpoint from the run.
    """
    artifact_uri = mlflow.get_run(run_id).to_dictionary()["info"]["artifact_uri"]
    artifact_dir = urlparse(artifact_uri).path
    checkpoint_dirs = sorted([f for f in os.listdir(artifact_dir) if f.startswith("checkpoint_")])
    best_checkpoint_dir = checkpoint_dirs[-2] if len(checkpoint_dirs) > 1 else checkpoint_dirs[-1]
    best_checkpoint = Checkpoint.from_directory(path=Path(artifact_dir, best_checkpoint_dir))
    return best_checkpoint