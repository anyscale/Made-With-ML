import numpy as np
import pandas as pd
import pytest
import ray

from madewithml import data

@pytest.fixture(scope="module")
def df():
    data = [{"title": "a0", "description": "b0", "tag": "c0"}]
    df = pd.DataFrame(data)
    return df

def test_load_data():
    num_samples = 10
    ds = data.load_data(num_samples=num_samples)
    assert ds.count() == num_samples


def test_stratify_split():
    n_per_class = 10
    targets = n_per_class*["c1"] + n_per_class*["c2"]
    ds = ray.data.from_items([dict(target=t) for t in targets])
    train_ds, test_ds = data.stratify_split(ds, stratify="target", test_size=0.5)
    train_target_counts = train_ds.to_pandas().target.value_counts().to_dict()
    test_target_counts = test_ds.to_pandas().target.value_counts().to_dict()
    assert train_target_counts == test_target_counts


@pytest.mark.parametrize(
    "text, stopwords, cleaned_text",
    [
        ("Hello world", [], "hello world"),
        ("Hello world", ["world"], "hello"),
        ("Hello worlds", ["world"], "hello worlds"),
    ],
)
def test_clean_text(text, stopwords, cleaned_text):
    assert (
        data.clean_text(
            text=text,
            stopwords=stopwords,
        )
        == cleaned_text
    )


def test_preprocess(df):
    assert "text" not in df.columns
    df = data.preprocess(df)
    assert df.columns.tolist() == ["text", "tag"]


def test_tokenize(df):
    df = data.preprocess(df)
    in_batch = {col: df[col].to_numpy() for col in df.columns}
    out_batch = data.tokenize(in_batch)
    assert set(out_batch) == {"ids", "masks", "targets"}


def test_to_one_hot():
    in_batch = {"targets": [1]}
    out_batch = data.to_one_hot(in_batch, num_classes=3)
    assert np.array_equal(out_batch["targets"], [[0., 1., 0.]])

