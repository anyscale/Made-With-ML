# madewithml/evaluate.py
from typing import Dict

import numpy as np
import ray
import ray.train.torch  # NOQA: F401 (imported but unused)
from sklearn.metrics import precision_recall_fscore_support

from madewithml import utils


def evaluate_ds(
    ds: ray.data.Dataset, predictor: ray.train.torch.torch_predictor.TorchPredictor
) -> Dict:
    """Evaluate a model's performance on a labeled dataset.

    Args:
        ds (ray.data.Dataset): Ray Dataset with labels.
        predictor (ray.train.torch.torch_predictor.TorchPredictor): Ray Predictor from a checkpoint.

    Returns:
        Dict: model's performance metrics on the dataset.
    """
    # y_true
    preprocessor = predictor.get_preprocessor()
    targets = utils.get_values(preprocessor.transform(ds), col="targets")
    y_true = np.array(targets).argmax(1)

    # y_pred
    z = predictor.predict(data=ds.to_pandas())["predictions"]
    y_pred = z.argmax(1)

    # Evaluate
    metrics = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    performance = {"precision": metrics[0], "recall": metrics[1], "f1": metrics[2]}
    return performance
