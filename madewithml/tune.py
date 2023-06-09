import datetime
import json

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

from madewithml import data, train, utils
from madewithml.config import MLFLOW_TRACKING_URI, logger

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
    experiment_name: str = typer.Option(..., "--experiment-id", help="name of the experiment for this training workload"),
    dataset_loc: str = typer.Option(..., "--dataset-loc", help="location of the dataset"),
    num_repartitions: int = typer.Option(..., "--num-repartitions", help="number of repartitions to use for the dataset"),
    initial_params: str = typer.Option(..., "--initial-params", help="initial set of parameters to use for tuning"),
    num_workers: int = typer.Option(1, "--num-workers", help="number of workers to use for training"),
    cpu_per_worker: int = typer.Option(1, "--cpu-per-worker", help="number of CPUs to use per worker"),
    gpu_per_worker: int = typer.Option(0, "--gpu-per-worker", help="number of GPUs to use per worker"),
    num_runs: int = typer.Option(0, "--num-runs", help="number of runs in this tuning experiment"),
    num_samples: int = typer.Option(None, "--num-samples", help="number of samples to use from dataset"),
    num_epochs: int = typer.Option(..., "--num-epochs", help="number of epochs to train for"),
    batch_size: int = typer.Option(..., "--batch-size", help="number of samples per batch"),
    results_fp: str = typer.Option(None, "--results-fp", help="filepath to save results to"),
) -> ray.tune.result_grid.ResultGrid:
    """Hyperparameter tuning experiment.

    Args:
        experiment_name (str): name of the experiment for this training workload.
        dataset_loc (str): location of the dataset.
        num_repartitions (int): number of repartitions to use for the dataset.
        initial_params (str): initial config for the tuning workload.
        num_workers (int, optional): number of workers to use for training. Defaults to 1.
        cpu_per_worker (int, optional): number of CPUs to use per worker. Defaults to 1.
        gpu_per_worker (int, optional): number of GPUs to use per worker. Defaults to 0.
        num_runs (int, optional): number of runs in this tuning experiment. Defaults to 1.
        num_samples (int, optional): number of samples to use from dataset.
            If this is passed in, it will override the config. Defaults to None.
        num_epochs (int, optional): number of epochs to train for.
            If this is passed in, it will override the config. Defaults to None.
        batch_size (int, optional): number of samples per batch.
            If this is passed in, it will override the config. Defaults to None.
        results_fp (str, optional): filepath to save the tuning results. Defaults to None.

    Returns:
        ray.tune.result_grid.ResultGrid: results of the tuning experiment.
    """
    # Set up
    utils.set_seeds()
    train_loop_config = {}
    train_loop_config["num_samples"] = num_samples
    train_loop_config["num_epochs"] = num_epochs
    train_loop_config["batch_size"] = batch_size

    # Scaling config
    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=int(bool(gpu_per_worker)),
        resources_per_worker={"CPU": cpu_per_worker, "GPU": gpu_per_worker},
        _max_cpu_fraction_per_node=0.8,
    )

    # Dataset
    ds = data.load_data(
        dataset_loc=dataset_loc,
        num_samples=train_loop_config.get("num_samples", None),
        num_partitions=num_repartitions,
    )
    train_ds, val_ds = data.stratify_split(ds, stratify="tag", test_size=0.2)

    # Dataset config
    dataset_config = {
        "train": DatasetConfig(fit=False, transform=False, randomize_block_order=False),
        "val": DatasetConfig(fit=False, transform=False, randomize_block_order=False),
    }

    # Preprocess
    preprocessor = data.get_preprocessor()
    train_ds = preprocessor.fit_transform(train_ds)
    val_ds = preprocessor.transform(val_ds)
    train_ds = train_ds.materialize()
    val_ds = val_ds.materialize()

    # Trainer
    trainer = TorchTrainer(
        train_loop_per_worker=train.train_loop_per_worker,
        train_loop_config=train_loop_config,
        scaling_config=scaling_config,
        datasets={"train": train_ds, "val": val_ds},
        dataset_config=dataset_config,
        preprocessor=preprocessor,
    )

    # Checkpoint configuration
    checkpoint_config = CheckpointConfig(
        num_to_keep=1,
        checkpoint_score_attribute="val_loss",
        checkpoint_score_order="min",
    )
    stopping_criteria = {"training_iteration": train_loop_config["num_epochs"]}  # auto incremented at every train step

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
    initial_params = json.loads(initial_params)
    search_alg = HyperOptSearch(points_to_evaluate=initial_params)
    search_alg = ConcurrencyLimiter(search_alg, max_concurrent=2)  # trade off b/w optimization and search space

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
    best_trial = results.get_best_result(metric="val_loss", mode="min")
    d = {
        "timestamp": datetime.datetime.now().strftime("%B %d, %Y %I:%M:%S %p"),
        "run_id": utils.get_run_id(experiment_name=experiment_name, trial_id=best_trial.metrics["trial_id"]),
        "params": best_trial.config["train_loop_config"],
        "metrics": utils.dict_to_list(best_trial.metrics_dataframe.to_dict(), keys=["epoch", "train_loss", "val_loss"]),
    }
    logger.info(json.dumps(d, indent=2))
    if results_fp:  # pragma: no cover, saving results
        utils.save_dict(d, results_fp)
    return results


if __name__ == "__main__":  # pragma: no cover, application
    if ray.is_initialized():
        ray.shutdown()
    ray.init()
    app()
