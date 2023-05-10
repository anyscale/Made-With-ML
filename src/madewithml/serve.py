import argparse

import pandas as pd
import ray
from ray import serve
from ray.train.torch import TorchPredictor
from starlette.requests import Request

from config import config  # NOQA: F401 (imported but unused)
from madewithml import predict


@serve.deployment
class Model:
    def __init__(self, run_id):
        best_checkpoint = predict.get_best_checkpoint(run_id=run_id, metric="val_loss", direction="min")
        self.predictor = TorchPredictor.from_checkpoint(best_checkpoint)
        self.label_encoder = self.predictor.get_preprocessor().preprocessors[1]
        self.index_to_class = {v: k for k, v in self.label_encoder.stats_["unique_values(tag)"].items()}

    def __call__(self, title, description):
        df = pd.DataFrame([{"title": title, "description": description, "tag": "other"}])
        results = predict.predict_with_probs(df=df, predictor=self.predictor, index_to_class=self.index_to_class)
        return results


@serve.deployment
class CustomLogic:
    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold

    async def __call__(self, request: Request):
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
    parser.add_argument(
        "--threshold", type=float, default=0.8, help="threshold for a confident prediction (default: 0.8))"
    )
    args = parser.parse_args()
    ray.init(address="auto")
    serve.run(CustomLogic.bind(model=Model.bind(run_id=args.run_id), threshold=args.threshold))
