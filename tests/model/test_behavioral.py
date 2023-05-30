import numpy as np
import pandas as pd
import pytest
from ray.train.torch.torch_predictor import TorchPredictor

from madewithml import predict


@pytest.fixture(scope="module")
def predictor(run_id):
    best_checkpoint = predict.get_best_checkpoint(run_id=run_id, metric="val_loss", mode="min")
    predictor = TorchPredictor.from_checkpoint(best_checkpoint)
    return predictor


@pytest.fixture(scope="module")
def index_to_class(predictor):
    label_encoder = predictor.get_preprocessor().preprocessors[1]
    class_to_index = label_encoder.stats_["unique_values(tag)"]
    index_to_class = {v: k for k, v in class_to_index.items()}
    return index_to_class


@pytest.fixture(scope="module")
def get_label_fixture(predictor, index_to_class):
    def get_label(text):
        df = pd.DataFrame({"title": [text], "description": "", "tag": "other"})
        z = predictor.predict(data=df)["predictions"]
        label = predict.decode(np.stack(z).argmax(1), index_to_class)[0]
        return label

    return get_label


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
def test_invariance(input_a, input_b, label, get_label_fixture):
    """INVariance via verb injection (changes should not affect outputs)."""
    label_a = get_label_fixture(input_a)
    label_b = get_label_fixture(input_b)
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
def test_directional(input, label, get_label_fixture):
    """DIRectional expectations (changes with known outputs)."""
    prediction = get_label_fixture(text=input)
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
def test_mft(input, label, get_label_fixture):
    """Minimum Functionality Tests (simple input/output pairs)."""
    prediction = get_label_fixture(text=input)
    assert label == prediction
