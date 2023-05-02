from madewithml import tune

def test_tune_models():
    num_runs = 2
    results = tune.tune_models(
        experiment_name="test",
        num_runs=num_runs,
        num_samples=32,
        num_epochs=1,
        batch_size=32
    )
    assert len(results.get_dataframe()) == num_runs
