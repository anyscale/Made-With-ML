import argparse

import pandas as pd
import ray
from ray import serve
from ray.train.torch import TorchPredictor
from starlette.requests import Request

from config import config  # NOQA: F401 (imported but unused)
from madewithml import predict


@serve.deployment(route_prefix="/")
class FinetunedLLMDeployment:  # pragma: no cover, tested with inference workload
    def __init__(self, run_id):
        best_checkpoint = predict.get_best_checkpoint(run_id=run_id, metric="val_loss", direction="min")
        self.predictor = TorchPredictor.from_checkpoint(best_checkpoint)
        self.label_encoder = self.predictor.get_preprocessor().preprocessors[1]
        self.index_to_class = {v: k for k, v in self.label_encoder.stats_["unique_values(tag)"].items()}

    async def __call__(self, request: Request):
        title = request.query_params["title"]
        description = request.query_params["description"]
        df = pd.DataFrame([{"title": title, "description": description, "tag": "other"}])
        results = predict.predict_with_probs(df=df, predictor=self.predictor, index_to_class=self.index_to_class)
        return {"results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", help="run ID to use for serving.")
    args = parser.parse_args()
    ray.init(address="auto")
    serve.run(FinetunedLLMDeployment.bind(run_id=args.run_id))
