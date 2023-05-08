import tempfile
from pathlib import Path

import numpy as np
import ray
import torch

from madewithml import utils


def test_set_seed():
    utils.set_seeds()
    a = np.random.randn(2, 3)
    b = np.random.randn(2, 3)
    utils.set_seeds()
    x = np.random.randn(2, 3)
    y = np.random.randn(2, 3)
    assert np.array_equal(a, x)
    assert np.array_equal(b, y)


def test_save_and_load_dict():
    with tempfile.TemporaryDirectory() as dp:
        d = {"hello": "world"}
        fp = Path(dp, "d.json")
        utils.save_dict(d=d, path=fp)
        d = utils.load_dict(path=fp)
        assert d["hello"] == "world"


def test_get_values():
    ds = ray.data.from_items([{"a": 1, "b": 2}])
    assert utils.get_values(ds, "a") == np.array(1)


def test_pad_array():
    arr = np.array([[1, 2], [1, 2, 3]], dtype="object")
    padded_arr = np.array([[1, 2, 0], [1, 2, 3]])
    assert np.array_equal(utils.pad_array(arr), padded_arr)


def test_collate_fn():
    batch = {"ids": np.array([[1, 2], [1, 2, 3]], dtype="object")}
    processed_batch = utils.collate_fn(batch)
    expected_batch = {"ids": torch.tensor([[1, 2, 0], [1, 2, 3]], dtype=torch.int32)}
    for k in batch:
        print(processed_batch[k], expected_batch[k])
        assert torch.allclose(processed_batch[k], expected_batch[k])
