import ray
import typer
from ray import tune
from ray.air.config import (
    CheckpointConfig,
    DatasetConfig,
    RunConfig,
    ScalingConfig,
)
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.train.torch import TorchTrainer
from ray.tune import Tuner
from ray.tune.experiment import Trial
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.hyperopt import HyperOptSearch

from config.config import CONFIG_FP, MLFLOW_TRACKING_URI
from madewithml import data, train, utils

# Initialize Typer CLI app
app = typer.Typer()


# Fixing https://github.com/ray-project/ray/blob/3aa6ede43743a098b5e0eb37ec11505f46100313/python/ray/air/integrations/mlflow.py#L301
class MLflowLoggerCallbackFixed(MLflowLoggerCallback):  # pragma: no cover, tested in larger tune workload
    def log_trial_start(self, trial: "Trial"):
        if trial not in self._trial_runs:
            tags = self.tags.copy()
            tags["trial_name"] = str(trial)
            run = self.mlflow_util.start_run(tags=tags, run_name=str(trial))
            self._trial_runs[trial] = run.info.run_id
        run_id = self._trial_runs[trial]
        config = trial.config
        self.mlflow_util.log_params(run_id=run_id, params_to_log=config["train_loop_config"])


@app.command()
def tune_models(
    experiment_name: str,
    use_gpu: bool = False,
    num_cpu_workers: int = 1,
    num_gpu_workers: int = 1,
    num_runs: int = 1,
    num_samples: int = None,
    num_epochs: int = None,
    batch_size: int = None,
) -> ray.tune.result_grid.ResultGrid:
    """Hyperparameter tuning experiment.

    Args:
        experiment_name (str): name of the experiment for this training workload.
        use_gpu (bool, optional): whether or not to use the GPU for training. Defaults to False.
        num_cpu_workers (int, optional): number of cpu workers to use for
            distributed data processing (and training if `use_gpu` is false). Defaults to 1.
        num_gpu_workers (int, optional): number of gpu workers to use for
                training (if `use_gpu` is false). Defaults to 1.
        num_runs (int, optional): number of runs in this tuning experiment. Defaults to 1.
        num_samples (int, optional): number of samples to use from dataset.
            If this is passed in, it will override the config. Defaults to None.
        num_epochs (int, optional): number of epochs to train for.
            If this is passed in, it will override the config. Defaults to None.
        batch_size (int, optional): number of samples per batch.
            If this is passed in, it will override the config. Defaults to None.

    Returns:
        ray.tune.result_grid.ResultGrid: results of the tuning experiment.
    """
    # Set up
    utils.set_seeds()
    train_loop_config = utils.load_dict(path=CONFIG_FP)
    train_loop_config["device"] = "cpu" if not use_gpu else "cuda"
    train_loop_config["num_samples"] = (
        num_samples if num_samples else train_loop_config["num_samples"]
    )
    train_loop_config["num_epochs"] = num_epochs if num_epochs else train_loop_config["num_epochs"]
    train_loop_config["batch_size"] = batch_size if batch_size else train_loop_config["batch_size"]

    # Dataset
    ds = data.load_data(
        num_samples=train_loop_config["num_samples"], num_partitions=num_cpu_workers
    )
    train_ds, val_ds = data.stratify_split(ds, stratify="tag", test_size=0.2)
    dataset_config = {
        "train": DatasetConfig(randomize_block_order=False),
        "val": DatasetConfig(randomize_block_order=False),
    }

    # Scaling config
    scaling_config = ScalingConfig(
        num_workers=num_gpu_workers if use_gpu else num_cpu_workers,
        use_gpu=use_gpu,
        _max_cpu_fraction_per_node=0.8,
    )

    # Trainer
    trainer = TorchTrainer(
        train_loop_per_worker=train.train_loop_per_worker,
        train_loop_config=train_loop_config,
        scaling_config=scaling_config,
        datasets={"train": train_ds, "val": val_ds},
        dataset_config=dataset_config,
        preprocessor=data.get_preprocessor(),
    )

    # Checkpoint configuration
    checkpoint_config = CheckpointConfig(
        num_to_keep=1, checkpoint_score_attribute="val_loss", checkpoint_score_order="min"
    )
    stopping_criteria = {
        "training_iteration": train_loop_config["num_epochs"]
    }  # auto incremented at every train step

    # Run configuration
    mlflow_callback = MLflowLoggerCallbackFixed(
        tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=experiment_name,
        save_artifact=True,
    )
    run_config = RunConfig(
        callbacks=[mlflow_callback],
        checkpoint_config=checkpoint_config,
        stop=stopping_criteria,
    )

    # Hyperparameters to start with
    initial_params = [
        {"train_loop_config": {"dropout_p": 0.5, "lr": 1e-4, "lr_factor": 0.8, "lr_patience": 3}}
    ]
    search_alg = HyperOptSearch(points_to_evaluate=initial_params)
    search_alg = ConcurrencyLimiter(
        search_alg, max_concurrent=2
    )  # trade off b/w optimization and search space

    # Parameter space
    param_space = {
        "train_loop_config": {
            "dropout_p": tune.uniform(0.3, 0.9),
            "lr": tune.loguniform(1e-5, 5e-4),
            "lr_factor": tune.uniform(0.1, 0.9),
            "lr_patience": tune.uniform(1, 10),
        }
    }

    # Stopping criteria
    scheduler = AsyncHyperBandScheduler(
        max_t=train_loop_config["num_epochs"],  # max epoch (<time_attr>) per trial
        grace_period=1,  # min epoch (<time_attr>) per trial
    )

    # Tune config
    tune_config = tune.TuneConfig(
        metric="val_loss",
        mode="min",
        search_alg=search_alg,
        scheduler=scheduler,
        num_samples=num_runs,
    )

    # Tuner
    tuner = Tuner(
        trainable=trainer,
        run_config=run_config,
        param_space=param_space,
        tune_config=tune_config,
    )

    # Tune
    results = tuner.fit()

    return results


if __name__ == "__main__":
    if ray.is_initialized():
        ray.shutdown()
    ray.init()
    app()
