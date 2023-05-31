import json

import pytest

from madewithml import train


@pytest.mark.training
def test_train_model(dataset_loc, generate_experiment_name, delete_experiment):
    experiment_name = generate_experiment_name(prefix="test_train")
    train_loop_config = {"dropout_p": 0.5, "lr": 1e-4, "lr_factor": 0.8, "lr_patience": 3}
    result = train.train_model(
        experiment_name=experiment_name,
        dataset_loc=dataset_loc,
        train_loop_config=json.dumps(train_loop_config),
        num_samples=256,
        num_epochs=2,
        batch_size=32,
    )
    delete_experiment(experiment_name=experiment_name)
    train_loss_list = result.metrics_dataframe.to_dict()["train_loss"]
    assert train_loss_list[0] > train_loss_list[1]  # loss decreased
