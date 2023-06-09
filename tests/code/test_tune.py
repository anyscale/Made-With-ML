import json

import pytest

from madewithml import tune


@pytest.mark.training
def test_tune_models(dataset_loc, generate_experiment_name, delete_experiment):
    num_runs = 2
    experiment_name = generate_experiment_name(prefix="test_tune")
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
        dataset_loc=dataset_loc,
        num_repartitions=3,
        initial_params=json.dumps(initial_params),
        num_workers=6,
        cpu_per_worker=1,
        gpu_per_worker=0,
        num_runs=num_runs,
        num_epochs=1,
        num_samples=512,
        batch_size=256,
        results_fp=None,
    )
    delete_experiment(experiment_name=experiment_name)
    assert len(results.get_dataframe()) == num_runs
