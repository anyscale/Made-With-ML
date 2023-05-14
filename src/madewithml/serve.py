import argparse
from typing import Dict, List, Union

import mlflow
import pandas as pd
import ray
from ray import serve
from ray.train.torch import TorchPredictor
from starlette.requests import Request

from madewithml import predict
from madewithml.config import MLFLOW_TRACKING_URI


@serve.deployment
class Model:
    """Train model that can be used for inference."""

    def __init__(self, run_id: str):
        """Initialize the model.

        Args:
            run_id (str): ID of the MLflow run to load the model from.
        """
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)  # so workers have access to model registry
        best_checkpoint = predict.get_best_checkpoint(run_id=run_id, metric="val_loss", direction="min")
        self.predictor = TorchPredictor.from_checkpoint(best_checkpoint)
        self.label_encoder = self.predictor.get_preprocessor().preprocessors[1]
        self.index_to_class = {v: k for k, v in self.label_encoder.stats_["unique_values(tag)"].items()}

    def __call__(self, title: str, description: str) -> List[Dict[str, Union[str, float]]]:
        """Make predictions given a title and description.

        Args:
            title (str): title of the project to classify.
            description (str): description of the project to classify.

        Returns:
            List[Dict[str, Union[str, float]]]: inference results.
        """
        df = pd.DataFrame([{"title": title, "description": description, "tag": "other"}])
        results = predict.predict_with_probs(df=df, predictor=self.predictor, index_to_class=self.index_to_class)
        return results


@serve.deployment
class CustomLogic:
    """Custom logic to apply to the model's predictions."""

    def __init__(self, model: Model, threshold: float):
        """Initialize the custom logic.

        Args:
            model (Model): model to apply custom logic to.
            threshold (float): threshold to replace prediction with `other` class.
        """
        self.model = model
        self.threshold = threshold

    async def __call__(self, request: Request) -> List[Dict[str, Union[str, float]]]:
        """Apply custom logic to the model's predictions.

        Args:
            request (Request): request object containing the title and description of the project to classify.

        Returns:
            List[Dict[str, Union[str, float]]]: inference results.
        """
        # Model's prediction
        data = await request.json()
        results_ref = await self.model.remote(title=data.get("title", ""), description=data.get("description", ""))
        results = await results_ref

        # Apply custom logic
        for i, result in enumerate(results):
            pred = result["prediction"]
            prob = result["probabilities"]
            if prob[pred] < self.threshold:
                results[i]["prediction"] = "other"

        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", help="run ID to use for serving.")
    parser.add_argument("--threshold", type=float, default=0.9, help="threshold for `other` class.")
    args = parser.parse_args()
    ray.init(address="auto")
    serve.run(CustomLogic.bind(model=Model.bind(run_id=args.run_id), threshold=args.threshold))
