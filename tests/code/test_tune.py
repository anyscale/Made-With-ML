import pytest
from utils import delete_experiment, generate_experiment_name

from madewithml import tune


@pytest.mark.training
def test_tune_models():
    num_runs = 2
    experiment_name = generate_experiment_name(prefix="test_tune")
    results = tune.tune_models(
        experiment_name=experiment_name,
        num_runs=num_runs,
        num_samples=32,
        num_epochs=1,
        batch_size=32,
    )
    delete_experiment(experiment_name=experiment_name)
    assert len(results.get_dataframe()) == num_runs
