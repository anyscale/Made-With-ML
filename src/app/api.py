# app/api.py
from http import HTTPStatus
from typing import Dict

import mlflow
import requests
from fastapi import FastAPI
from ray import serve
from starlette.requests import Request

from app.schemas import Project
from config import config  # NOQA: F401 (imported but unused)
from madewithml import predict
from madewithml.utils import construct_response

# Define app
app = FastAPI(
    title="Made With ML",
    description="Classify machine learning projects.",
    version="0.1",
)


@app.on_event("startup")
def get_run_id():
    global run_id
    experiment_name = "llm"
    sorted_runs = mlflow.search_runs(
        experiment_names=[experiment_name], order_by=["metrics.val_loss ASC"]
    )
    run_id = sorted_runs.iloc[0].run_id


@app.get("/", tags=["General"])
@construct_response
def _index(request: Request) -> Dict:
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    return response


@app.post("/predict", tags=["Prediction"])
@construct_response
def _predict(request: Request, payload: Project) -> Dict:
    """Predict tags for a list of projects."""
    results = predict.predict(title=payload.title, description=payload.description, run_id=run_id)
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"results": results},
    }
    return response


@serve.deployment(route_prefix="/")
@serve.ingress(app)
class FastAPIWrapper:
    pass


if __name__ == "__main__":
    serve.run(FastAPIWrapper.bind())
    headers = {"Content-Type": "application/json"}
    data = {
        "title": "Transfer learning with transformers",
        "description": "Using transformers for transfer learning on text classification tasks.",
    }
    results = requests.post("http://127.0.0.1:8000/predict", headers=headers, json=data)
    print(results.json())
