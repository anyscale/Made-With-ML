# app/api.py
import mlflow
import pandas as pd
import ray
from ray import serve
from ray.train.torch import TorchPredictor
from starlette.requests import Request

from config import config  # NOQA: F401 (imported but unused)
from madewithml import predict


@serve.deployment(route_prefix="/")
class FinetunedLLMDeployment:
    def __init__(self, run_id):
        best_checkpoint = predict.get_best_checkpoint(run_id=run_id)
        self.predictor = TorchPredictor.from_checkpoint(best_checkpoint)
        self.label_encoder = self.predictor.get_preprocessor().preprocessors[1]
        self.index_to_class = {
            v: k for k, v in self.label_encoder.stats_["unique_values(tag)"].items()
        }

    async def __call__(self, request: Request):
        title = request.query_params["title"]
        description = request.query_params["description"]
        df = pd.DataFrame([{"title": title, "description": description, "tag": "other"}])
        results = predict.predict_with_probs(
            df=df, predictor=self.predictor, index_to_class=self.index_to_class
        )
        return {"results": results}


if __name__ == "__main__":
    ray.init(address="auto")
    sorted_runs = mlflow.search_runs(experiment_names=["llm"], order_by=["metrics.val_loss ASC"])
    serve.run(FinetunedLLMDeployment.bind(run_id=sorted_runs.iloc[0].run_id))
