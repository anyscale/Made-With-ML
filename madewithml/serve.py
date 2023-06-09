import argparse
from http import HTTPStatus
from typing import Dict

import pandas as pd
import ray
from fastapi import FastAPI
from ray import serve
from ray.train.torch import TorchPredictor
from starlette.requests import Request

from madewithml import predict
from madewithml.config import MLFLOW_TRACKING_URI, mlflow

# Define application
app = FastAPI(
    title="Made With ML",
    description="Classify machine learning projects.",
    version="0.1",
)


@serve.deployment(route_prefix="/")
@serve.ingress(app)
class ModelDeployment:
    def __init__(self, run_id: str, threshold: int = 0.9):
        """Initialize the model."""
        self.run_id = run_id
        self.threshold = threshold
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)  # so workers have access to model registry
        best_checkpoint = predict.get_best_checkpoint(run_id=self.run_id, metric="val_loss", mode="min")
        self.predictor = TorchPredictor.from_checkpoint(best_checkpoint)
        self.label_encoder = self.predictor.get_preprocessor().preprocessors[1]
        self.index_to_class = {v: k for k, v in self.label_encoder.stats_["unique_values(tag)"].items()}

    @app.get("/")
    def _index(self) -> Dict:
        """Health check."""
        response = {
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
            "data": {},
        }
        return response

    @app.get("/run_id")
    def _run_id(self) -> Dict:
        """Get the run ID."""
        return {"run_id": self.run_id}

    @app.post("/predict")
    async def _predict(self, request: Request) -> Dict:
        # Get predictions
        data = await request.json()
        df = pd.DataFrame(
            [
                {
                    "title": data.get("title", ""),
                    "description": data.get("description", ""),
                    "tag": "other",
                }
            ]
        )
        results = predict.predict_with_probs(df=df, predictor=self.predictor, index_to_class=self.index_to_class)

        # Apply custom logic
        for i, result in enumerate(results):
            pred = result["prediction"]
            prob = result["probabilities"]
            if prob[pred] < self.threshold:
                results[i]["prediction"] = "other"

        return {"results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", help="run ID to use for serving.")
    parser.add_argument("--threshold", type=float, default=0.9, help="threshold for `other` class.")
    args = parser.parse_args()
    ray.init(address="auto")
    serve.run(ModelDeployment.bind(run_id=args.run_id, threshold=args.threshold))
