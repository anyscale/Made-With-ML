# madewithml/tune.py
from ray import tune
from ray.air.config import  RunConfig, ScalingConfig
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.train.torch import TorchTrainer
from ray.tune import Tuner
from ray.tune.schedulers import AsyncHyperBandScheduler
import typer

from config import config
from madewithml import train, utils


# Initialize Typer CLI app
app = typer.Typer()


# Fixing https://github.com/ray-project/ray/blob/3aa6ede43743a098b5e0eb37ec11505f46100313/python/ray/air/integrations/mlflow.py#L301
class MLflowLoggerCallbackFixed(MLflowLoggerCallback):
    def log_trial_start(self, trial: "Trial"):
        if trial not in self._trial_runs:
            tags = self.tags.copy()
            tags["trial_name"] = str(trial)
            run = self.mlflow_util.start_run(tags=tags, run_name=str(trial))
            self._trial_runs[trial] = run.info.run_id
        run_id = self._trial_runs[trial]
        config = trial.config
        self.mlflow_util.log_params(run_id=run_id, params_to_log=config["train_loop_config"])


def tune_models(experiment_name: str, num_runs: int = 10, num_workers: int = 1,
                use_gpu: bool = False, args_fp:str = config.ARGS_FP):
    """Hyperparameter tuning on many models."""
    # Scaling config
    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=use_gpu,
        _max_cpu_fraction_per_node=0.8,
    )

    # Trainer
    args = utils.load_dict(path=args_fp)
    trainer = TorchTrainer(
        train_loop_per_worker=train.training_loop,
        train_loop_config=args,
        scaling_config=scaling_config,
    )

    # Run configuration
    stopping_criteria = {"training_iteration": args["num_epochs"]}  # auto incremented at every train step
    run_config = RunConfig(
        callbacks=[MLflowLoggerCallbackFixed(
            tracking_uri=config.MLFLOW_TRACKING_URI,
            experiment_name=experiment_name,
            save_artifact=True)],
        stop=stopping_criteria
    )

    # Parameter space
    param_space = {
        "train_loop_config": {
            "dropout_p": tune.uniform(0.3, 0.7),
            "lr": tune.loguniform(1e-5, 1e-3)
        }
    }

    # Stopping criteria
    scheduler = AsyncHyperBandScheduler(
        max_t=args["num_epochs"],  # max epoch (<time_attr>) per trial
        grace_period=1,  # min epoch (<time_attr>) per trial
    )

    # Tune config
    tune_config = tune.TuneConfig(
        metric="val_loss",
        mode="min",
        num_samples=num_runs,
        scheduler=scheduler
    )

    # Tuner
    tuner = Tuner(
        trainable=trainer,
        run_config=run_config,
        param_space=param_space,
        tune_config=tune_config,
    )

    # Tune
    result_grid = tuner.fit()
    return result_grid


if __name__ == "__main__":
    app()