import json

import pytest
import utils as test_utils

from madewithml import train


@pytest.mark.training
def test_train_model():
    experiment_name = test_utils.generate_experiment_name(prefix="test_train")
    train_loop_config = {"dropout_p": 0.5, "lr": 1e-4, "lr_factor": 0.8, "lr_patience": 3}
    result = train.train_model(
        experiment_name=experiment_name,
        train_loop_config=json.dumps(train_loop_config),
        num_samples=32,
        num_epochs=2,
        batch_size=32,
    )
    test_utils.delete_experiment(experiment_name=experiment_name)
    train_loss_dict = result.metrics_dataframe.to_dict()["train_loss"]
    assert train_loss_dict[0] > train_loss_dict[1]  # loss decreased
