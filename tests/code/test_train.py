import pytest
from utils import delete_experiment, generate_experiment_name

from madewithml import train


@pytest.mark.training
def test_train_model():
    experiment_name = generate_experiment_name(prefix="test_train")
    result = train.train_model(experiment_name=experiment_name, num_samples=32, num_epochs=2, batch_size=32)
    delete_experiment(experiment_name=experiment_name)
    train_loss_dict = result.metrics_dataframe.to_dict()["train_loss"]
    assert train_loss_dict[0] > train_loss_dict[1]  # loss decreased
