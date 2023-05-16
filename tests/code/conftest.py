import uuid

import pytest

from madewithml.config import mlflow


@pytest.fixture
def dataset_loc():
    return "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/madewithml/dataset.csv"


@pytest.fixture
def generate_experiment_name(request):
    def f(prefix: str = "test") -> str:
        return f"{prefix}-{uuid.uuid4().hex[:8]}"

    return f


@pytest.fixture
def delete_experiment(request):
    def f(experiment_name: str) -> None:
        client = mlflow.tracking.MlflowClient()
        experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
        client.delete_experiment(experiment_id=experiment_id)

    return f
