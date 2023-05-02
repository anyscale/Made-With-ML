from madewithml import train

def test_train_model():
    result = train.train_model(
        experiment_name="test",
        num_samples=32,
        num_epochs=2,
        batch_size=32
    )
    train_loss_dict = result.metrics_dataframe.to_dict()["train_loss"]
    assert train_loss_dict[0] > train_loss_dict[1]  # loss decreased
