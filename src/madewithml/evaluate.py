import argparse
import json
from collections import OrderedDict
from typing import Dict

import numpy as np
import ray
import ray.train.torch  # NOQA: F401 (imported but unused)
from ray.data import Dataset
from ray.data.preprocessor import Preprocessor
from ray.train.torch.torch_predictor import TorchPredictor
from sklearn.metrics import precision_recall_fscore_support
from snorkel.slicing import PandasSFApplier, slicing_function

from madewithml import predict, utils
from madewithml.config import HOLDOUT_LOC, logger


def get_overall_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:  # pragma: no cover, eval workload
    """Get overall performance metrics.

    Args:
        y_true (np.ndarray): ground truth labels.
        y_pred (np.ndarray): predicted labels.

    Returns:
        Dict: overall metrics.
    """
    metrics = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    overall_metrics = {
        "precision": metrics[0],
        "recall": metrics[1],
        "f1": metrics[2],
        "num_samples": np.float64(len(y_true)),
    }
    return overall_metrics


def get_per_class_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_to_index: Dict
) -> Dict:  # pragma: no cover, eval workload
    """Get per class performance metrics.

    Args:
        y_true (np.ndarray): ground truth labels.
        y_pred (np.ndarray): predicted labels.
        class_to_index (Dict): dictionary mapping class to index.

    Returns:
        Dict: per class metrics.
    """
    per_class_metrics = {}
    metrics = precision_recall_fscore_support(y_true, y_pred, average=None)
    for i, _class in enumerate(class_to_index):
        per_class_metrics[_class] = {
            "precision": metrics[0][i],
            "recall": metrics[1][i],
            "f1": metrics[2][i],
            "num_samples": np.float64(metrics[3][i]),
        }
    sorted_per_class_metrics = OrderedDict(
        sorted(per_class_metrics.items(), key=lambda tag: tag[1]["f1"], reverse=True)
    )
    return sorted_per_class_metrics


@slicing_function()
def nlp_llm(x):  # pragma: no cover, eval workload
    """NLP projects that use LLMs."""
    nlp_project = "natural-language-processing" in x.tag
    llm_terms = ["transformer", "llm", "bert"]
    llm_project = any(s.lower() in x.text.lower() for s in llm_terms)
    return nlp_project and llm_project


@slicing_function()
def short_text(x):  # pragma: no cover, eval workload
    """Projects with short titles and descriptions."""
    return len(x.text.split()) < 8  # less than 8 words


def get_slice_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, ds: Dataset, preprocessor: Preprocessor
) -> Dict:  # pragma: no cover, eval workload
    """Get performance metrics for slices.

    Args:
        y_true (np.ndarray): ground truth labels.
        y_pred (np.ndarray): predicted labels.
        ds (Dataset): Ray dataset with labels.
        preprocessor (Preprocessor): Ray preprocessor.

    Returns:
        Dict: performance metrics for slices.
    """
    slice_metrics = {}
    df = preprocessor.preprocessors[0].transform(ds).to_pandas()
    slicing_functions = [nlp_llm, short_text]
    applier = PandasSFApplier(slicing_functions)
    slices = applier.apply(df)
    for slice_name in slices.dtype.names:
        mask = slices[slice_name].astype(bool)
        if sum(mask):
            metrics = precision_recall_fscore_support(y_true[mask], y_pred[mask], average="micro")
            slice_metrics[slice_name] = {}
            slice_metrics[slice_name]["precision"] = metrics[0]
            slice_metrics[slice_name]["recall"] = metrics[1]
            slice_metrics[slice_name]["f1"] = metrics[2]
            slice_metrics[slice_name]["num_samples"] = len(y_true[mask])
    return slice_metrics


def evaluate(ds: Dataset, predictor: TorchPredictor) -> Dict:  # pragma: no cover, eval workload
    """Evaluate a model's performance on a labeled dataset.

    Args:
        ds (Dataset): Ray Dataset with labels.
        predictor (TorchPredictor): Ray Predictor from a checkpoint.

    Returns:
        Dict: model's performance metrics on the dataset.
    """
    # y_true
    preprocessor = predictor.get_preprocessor()
    targets = utils.get_arr_col(preprocessor.transform(ds), col="targets")
    y_true = targets.argmax(1)

    # y_pred
    z = predictor.predict(data=ds.to_pandas())["predictions"]
    y_pred = np.stack(z).argmax(1)

    # Components
    label_encoder = preprocessor.preprocessors[1]
    class_to_index = label_encoder.stats_["unique_values(tag)"]

    # Metrics
    metrics = {
        "overall": get_overall_metrics(y_true=y_true, y_pred=y_pred),
        "per_class": get_per_class_metrics(y_true=y_true, y_pred=y_pred, class_to_index=class_to_index),
        "slices": get_slice_metrics(y_true=y_true, y_pred=y_pred, ds=ds, preprocessor=preprocessor),
    }
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", help="run ID to use for serving.")
    parser.add_argument("--num-cpu-workers", type=int, help="num of workers to process the dataset")
    args = parser.parse_args()

    # Evaluate
    ds = ray.data.read_csv(HOLDOUT_LOC).repartition(args.num_cpu_workers)
    best_checkpoint = predict.get_best_checkpoint(run_id=args.run_id, metric="val_loss", direction="min")
    predictor = TorchPredictor.from_checkpoint(best_checkpoint)
    metrics = evaluate(ds=ds, predictor=predictor)
    logger.info(json.dumps(metrics, indent=2))
