import tempfile
from pathlib import Path

import numpy as np
import ray

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
    assert utils.get_values(ds, "a") == [1]
