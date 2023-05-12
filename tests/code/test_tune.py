import json

import pytest
import utils as test_utils

from madewithml import tune


@pytest.mark.training
def test_tune_models():
    num_runs = 2
    experiment_name = test_utils.generate_experiment_name(prefix="test_tune")
    initial_params = [
        {
            "train_loop_config": {
                "dropout_p": 0.5,
                "lr": 1e-4,
                "lr_factor": 0.8,
                "lr_patience": 3,
            }
        }
    ]
    results = tune.tune_models(
        experiment_name=experiment_name,
        initial_params=json.dumps(initial_params),
        num_runs=num_runs,
        num_samples=32,
        num_epochs=1,
        batch_size=32,
    )
    test_utils.delete_experiment(experiment_name=experiment_name)
    assert len(results.get_dataframe()) == num_runs
