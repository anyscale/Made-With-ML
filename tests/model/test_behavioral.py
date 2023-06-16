import numpy as np
import pandas as pd
import pytest
from ray.train.torch.torch_predictor import TorchPredictor

from madewithml import predict


@pytest.fixture(scope="module")
def predictor(run_id):
    best_checkpoint = predict.get_best_checkpoint(run_id=run_id)
    predictor = TorchPredictor.from_checkpoint(best_checkpoint)
    return predictor


@pytest.fixture(scope="module")
def get_label(text):
    df = pd.DataFrame({"title": [text], "description": "", "tag": "other"})
    z = predictor.predict(data=df)["predictions"]
    preprocessor = predictor.get_preprocessor()
    label = predict.decode(np.stack(z).argmax(1), preprocessor.index_to_class)[0]
    return label


@pytest.mark.parametrize(
    "input_a, input_b, label",
    [
        (
            "Transformers applied to NLP have revolutionized machine learning.",
            "Transformers applied to NLP have disrupted machine learning.",
            "natural-language-processing",
        ),
    ],
)
def test_invariance(input_a, input_b, label, get_label):
    """INVariance via verb injection (changes should not affect outputs)."""
    label_a = get_label(input_a)
    label_b = get_label(input_b)
    assert label_a == label_b == label


@pytest.mark.parametrize(
    "input, label",
    [
        (
            "ML applied to text classification.",
            "natural-language-processing",
        ),
        (
            "ML applied to image classification.",
            "computer-vision",
        ),
        (
            "CNNs for text classification.",
            "natural-language-processing",
        ),
    ],
)
def test_directional(input, label, get_label):
    """DIRectional expectations (changes with known outputs)."""
    prediction = get_label(text=input)
    assert label == prediction


@pytest.mark.parametrize(
    "input, label",
    [
        (
            "Natural language processing is the next big wave in machine learning.",
            "natural-language-processing",
        ),
        (
            "MLOps is the next big wave in machine learning.",
            "mlops",
        ),
        (
            "This is about graph neural networks.",
            "other",
        ),
    ],
)
def test_mft(input, label, get_label):
    """Minimum Functionality Tests (simple input/output pairs)."""
    prediction = get_label(text=input)
    assert label == prediction
