# madewithml/predict.py
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List
from urllib.parse import urlparse

import mlflow
import pandas as pd
import ray
import torch
import typer
from numpyencoder import NumpyEncoder
from ray.air import Checkpoint
from ray.train.torch import TorchPredictor

from config.config import logger  # also needed to set MLflow URI

# Initialize Typer CLI app
app = typer.Typer()


def decode(indices: Iterable[Any], index_to_class: Dict) -> List:
    """Decode indices to labels.

    Args:
        indices (Iterable[Any]): Iterable (list, array, etc.) with indices.
        index_to_class (Dict): mapping between indices and labels.

    Returns:
        List: list of labels.
    """
    return [index_to_class[index] for index in indices]


def format_prob(prob: Iterable, index_to_class: Dict) -> Dict:
    """Format probabilities to a dictionary mapping class label to probability.

    Args:
        prob (Iterable): probabilities.
        index_to_class (Dict): mapping between indices and labels.

    Returns:
        Dict: Dictionary mapping class label to probability.
    """
    d = {}
    for i, item in enumerate(prob):
        d[index_to_class[i]] = item
    return d


def predict_with_probs(
    df: pd.DataFrame,
    predictor: ray.train.torch.torch_predictor.TorchPredictor,
    index_to_class: Dict,
) -> List:
    """Predict tags (with probabilities) for input data from a dataframe.

    Args:
        df (pd.DataFrame): dataframe with input features.
        predictor (ray.train.torch.torch_predictor.TorchPredictor): loaded predictor from a checkpoint.
        index_to_class (Dict): mapping between indices and labels.

    Returns:
        List: list of predicted labels.
    """
    z = predictor.predict(data=df)["predictions"]
    y_prob = torch.tensor(z).softmax(dim=1).numpy()
    results = []
    for i, prob in enumerate(y_prob):
        tag = decode([z[i].argmax()], index_to_class)
        results.append({"pred": tag, "prob": format_prob(prob, index_to_class)})
    return results


def get_best_checkpoint(run_id: str) -> ray.train.torch.torch_checkpoint.TorchCheckpoint:
    """Get the best checkpoint (by performance) from a specific run.

    Args:
        run_id (str): ID of the run to get the best checkpoint from.

    Returns:
        ray.train.torch.torch_checkpoint.TorchCheckpoint: Best `TorchCheckpoint` from the run.
    """
    artifact_uri = mlflow.get_run(run_id).to_dictionary()["info"]["artifact_uri"]
    artifact_dir = urlparse(artifact_uri).path
    checkpoint_dirs = sorted([f for f in os.listdir(artifact_dir) if f.startswith("checkpoint_")])
    best_checkpoint_dir = checkpoint_dirs[-2] if len(checkpoint_dirs) > 1 else checkpoint_dirs[-1]
    best_checkpoint = Checkpoint.from_directory(path=Path(artifact_dir, best_checkpoint_dir))
    return best_checkpoint


@app.command()
def predict(
    title: str = "", description: str = "", experiment_name: str = None, run_id: str = None
) -> Dict:
    """Predict the tag for a project given it's title and description.

    Args:
        title (str, optional): project title. Defaults to "".
        description (str, optional): project description. Defaults to "".
        experiment_name (str, optional): name of the experiment (used to find best
            `run_id` if it's not provided. Defaults to None.
        run_id (str, optional): id of the specific run to load from. Defaults to None.

    Returns:
        Dict: prediction results for the input data.
    """
    # Run ID
    if not run_id:
        sorted_runs = mlflow.search_runs(
            experiment_names=[experiment_name], order_by=["metrics.val_loss ASC"]
        )
        run_id = sorted_runs.iloc[0].run_id

    # Load components
    best_checkpoint = get_best_checkpoint(run_id=run_id)
    predictor = TorchPredictor.from_checkpoint(best_checkpoint)
    label_encoder = predictor.get_preprocessor().preprocessors[1]
    index_to_class = {v: k for k, v in label_encoder.stats_["unique_values(tag)"].items()}

    # Predict
    sample_df = pd.DataFrame([{"title": title, "description": description, "tag": "other"}])
    results = predict_with_probs(df=sample_df, predictor=predictor, index_to_class=index_to_class)
    logger.info(json.dumps(results, cls=NumpyEncoder, indent=2))
    return results


if __name__ == "__main__":
    app()
