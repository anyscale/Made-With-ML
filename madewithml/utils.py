import json
import os
import random
from typing import Any, Dict, List

import numpy as np
import torch
from ray.air._internal.torch_utils import (
    convert_ndarray_batch_to_torch_tensor_batch,
    get_device,
)
from ray.data import Dataset

from madewithml.config import mlflow


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
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):  # pragma: no cover
        os.makedirs(directory)
    with open(path, "w") as fp:
        json.dump(d, indent=2, fp=fp, cls=cls, sort_keys=sortkeys)
        fp.write("\n")


def get_arr_col(ds: Dataset, col: str) -> np.ndarray:
    """Return an array of values from a specific array column in a Ray Dataset.

    Args:
        ds (Dataset): Ray Dataset.
        col (str): name of the column to extract values from.

    Returns:
        np.array: an array of the column's values.
    """
    values = ds.map_batches(lambda batch: {col: batch[col]}, batch_format="numpy")
    return np.stack([item[col] for item in values.take_all()])


def pad_array(arr: np.ndarray, dtype=np.int32) -> np.ndarray:
    """Pad an 2D array with zeros until all rows in the
    2D array are of the same length as a the longest
    row in the 2D array.

    Args:
        arr (np.array): input array

    Returns:
        np.array: zero padded array
    """
    max_len = max(len(row) for row in arr)
    padded_arr = np.zeros((arr.shape[0], max_len), dtype=dtype)
    for i, row in enumerate(arr):
        padded_arr[i][: len(row)] = row
    return padded_arr


def collate_fn(batch: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    """Convert a batch of numpy arrays to tensors.

    Args:
        batch (Dict[str, np.ndarray]): input batch as a dictionary of numpy arrays.

    Returns:
        Dict[str, torch.Tensor]: output batch as a dictionary of tensors.
    """
    for k, v in batch.items():
        batch[k] = pad_array(v)
    dtypes = {"ids": torch.int32, "masks": torch.int32, "targets": torch.float64}
    return convert_ndarray_batch_to_torch_tensor_batch(batch, dtypes=dtypes, device=get_device())


def get_run_id(experiment_name: str, trial_id: str) -> str:  # pragma: no cover, mlflow functionality
    """Get the MLflow run ID for a specific Ray trial ID.

    Args:
        experiment_name (str): name of the experiment.
        trial_id (str): id of the trial.

    Returns:
        str: run id of the trial.
    """
    trial_name = f"TorchTrainer_{trial_id}"
    run = mlflow.search_runs(experiment_names=[experiment_name], filter_string=f"tags.trial_name = '{trial_name}'").iloc[0]
    return run.run_id


def dict_to_list(data: Dict, keys: List[str]) -> List[Dict[str, Any]]:
    """Convert a dictionary to a list of dictionaries.

    Args:
        data (Dict): input dictionary.
        keys (List[str]): keys to include in the output list of dictionaries.

    Returns:
        List[Dict[str, Any]]: output list of dictionaries.
    """
    list_of_dicts = []
    for i in range(len(data[keys[0]])):
        new_dict = {key: data[key][i] for key in keys}
        list_of_dicts.append(new_dict)
    return list_of_dicts
