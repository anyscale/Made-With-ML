import json
import os
import random
from typing import Any, Dict

import numpy as np
import torch
from ray.air._internal.torch_utils import (
    convert_ndarray_batch_to_torch_tensor_batch,
    get_device,
)
from ray.data import Dataset


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


def get_values(ds: Dataset, col: str) -> np.ndarray:
    """Return a list of values from a specific column in a Ray Dataset.

    Args:
        ds (Dataset): Ray Dataset.
        col (str): name of the column to extract values from.

    Returns:
        np.array: an array of the column's values.
    """
    return np.stack(ds.select_columns([col]).to_pandas()[col])


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
