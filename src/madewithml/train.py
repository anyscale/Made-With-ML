import datetime
import json
from typing import Tuple

import numpy as np
import ray
import ray.train as train
import torch
import torch.nn as nn
import typer
from ray.air import session
from ray.air._internal.torch_utils import get_device
from ray.air.config import (
    CheckpointConfig,
    DatasetConfig,
    RunConfig,
    ScalingConfig,
)
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.data import Dataset
from ray.train.torch import TorchCheckpoint, TorchTrainer
from transformers import BertModel

from madewithml import data, models, utils
from madewithml.config import ACCEPTED_TAGS, MLFLOW_TRACKING_URI, logger

# Initialize Typer CLI app
app = typer.Typer()


def train_step(
    ds: Dataset,
    batch_size: int,
    model: nn.Module,
    loss_fn: torch.nn.modules.loss._WeightedLoss,
    optimizer: torch.optim.Optimizer,
) -> float:  # pragma: no cover, tested via train workload
    """Train step.

    Args:
        ds (Dataset): dataset to iterate batches from.
        batch_size (int): size of each batch.
        model (nn.Module): model to train.
        loss_fn (torch.nn.loss._WeightedLoss): loss function to use between labels and predictions.
        optimizer (torch.optimizer.Optimizer): optimizer to use for updating the model's weights.

    Returns:
        float: cumulative loss for the dataset.
    """
    model.train()
    loss = 0.0
    ds_generator = ds.iter_torch_batches(batch_size=batch_size, collate_fn=utils.collate_fn)
    for i, batch in enumerate(ds_generator):
        optimizer.zero_grad()  # reset gradients
        z = model(batch)  # forward pass
        J = loss_fn(z, batch["targets"])  # define loss
        J.backward()  # backward pass
        optimizer.step()  # update weights
        loss += (J.detach().item() - loss) / (i + 1)  # cumulative loss
    return loss


def eval_step(
    ds: Dataset, batch_size: int, model: nn.Module, loss_fn: torch.nn.modules.loss._WeightedLoss
) -> Tuple[float, np.array, np.array]:  # pragma: no cover, tested via train workload
    """Eval step.

    Args:
        ds (Dataset): dataset to iterate batches from.
        batch_size (int): size of each batch.
        model (nn.Module): model to train.
        loss_fn (torch.nn.loss._WeightedLoss): loss function to use between labels and predictions.
        device (str, optional): which device (cpu or cuda) to run on.

    Returns:
        Tuple[float, np.array, np.array]: cumulative loss, ground truths and predictions.
    """
    model.eval()
    loss = 0.0
    y_trues, y_preds = [], []
    ds_generator = ds.iter_torch_batches(batch_size=batch_size, collate_fn=utils.collate_fn)
    with torch.inference_mode():
        for i, batch in enumerate(ds_generator):
            z = model(batch)
            J = loss_fn(z, batch["targets"]).item()
            loss += (J - loss) / (i + 1)
            y_trues.extend(torch.argmax(batch["targets"], dim=1).cpu().numpy())
            y_preds.extend(torch.argmax(z, dim=1).cpu().numpy())
    return loss, np.vstack(y_trues), np.vstack(y_preds)


def train_loop_per_worker(
    config: dict,
) -> None:  # pragma: no cover, tested via train workload
    """Training loop that each worker will execute.

    Args:
        config (dict): arguments to use for training.
    """
    # Set up
    utils.set_seeds()

    # Hyperparameters
    dropout_p = config["dropout_p"]
    lr = config["lr"]
    lr_factor = config["lr_factor"]
    lr_patience = config["lr_patience"]
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]

    # Get datasets
    train_ds = session.get_dataset_shard("train")
    val_ds = session.get_dataset_shard("val")

    # Model
    llm = BertModel.from_pretrained("allenai/scibert_scivocab_uncased", return_dict=False)
    model = models.FinetunedLLM(
        llm=llm,
        dropout_p=dropout_p,
        embedding_dim=llm.config.hidden_size,
        num_classes=len(ACCEPTED_TAGS),
    )
    model = train.torch.prepare_model(model)

    # Class weights
    batch_counts = []
    for batch in train_ds.iter_torch_batches(batch_size=256, collate_fn=utils.collate_fn):
        batch_counts.append(np.bincount(batch["targets"].cpu().numpy().argmax(1)))
    counts = [sum(count) for count in zip(*batch_counts)]
    class_weights = np.array([1.0 / count for i, count in enumerate(counts)])
    class_weights_tensor = torch.Tensor(class_weights).to(get_device())

    # Training components
    loss_fn = nn.BCEWithLogitsLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=lr_patience
    )

    # Train
    batch_size_per_worker = batch_size // session.get_world_size()
    for epoch in range(num_epochs):
        # Step
        train_loss = train_step(train_ds, batch_size_per_worker, model, loss_fn, optimizer)
        val_loss, _, _ = eval_step(val_ds, batch_size_per_worker, model, loss_fn)
        scheduler.step(val_loss)

        # Checkpoint
        metrics = dict(
            epoch=epoch,
            lr=optimizer.param_groups[0]["lr"],
            train_loss=train_loss,
            val_loss=val_loss,
        )
        session.report(metrics, checkpoint=TorchCheckpoint.from_model(model=model))


@app.command()
def train_model(
    experiment_name: str,
    train_loop_config: str,
    use_gpu: bool = False,
    num_cpu_workers: int = 1,
    num_gpu_workers: int = 1,
    num_samples: int = None,
    num_epochs: int = None,
    batch_size: int = None,
    results_fp: str = None,
) -> ray.air.result.Result:
    """Main train function to train our model as a distributed workload.

    Args:
        experiment_name (str): name of the experiment for this training workload.
        train_loop_config (str): arguments to use for training.
        use_gpu (bool, optional): whether or not to use the GPU for training. Defaults to False.
        num_cpu_workers (int, optional): number of cpu workers to use for
            distributed data processing (and training if `use_gpu` is false). Defaults to 1.
        num_gpu_workers (int, optional): number of gpu workers to use for
                training (if `use_gpu` is false). Defaults to 1.
        num_samples (int, optional): number of samples to use from dataset.
            If this is passed in, it will override the config. Defaults to None.
        num_epochs (int, optional): number of epochs to train for.
            If this is passed in, it will override the config. Defaults to None.
        batch_size (int, optional): number of samples per batch.
            If this is passed in, it will override the config. Defaults to None.
        results_fp (str, optional): filepath to save results to. Defaults to None.

    Returns:
        ray.air.result.Result: training results.
    """
    # Set up
    train_loop_config = json.loads(train_loop_config)
    train_loop_config["device"] = "cpu" if not use_gpu else "cuda"
    train_loop_config["num_samples"] = num_samples
    train_loop_config["num_epochs"] = num_epochs
    train_loop_config["batch_size"] = batch_size

    # Scaling config
    scaling_config = ScalingConfig(
        num_workers=num_gpu_workers if use_gpu else num_cpu_workers,
        use_gpu=use_gpu,
        _max_cpu_fraction_per_node=0.8,
    )

    # Checkpoint config
    checkpoint_config = CheckpointConfig(
        num_to_keep=1,
        checkpoint_score_attribute="val_loss",
        checkpoint_score_order="min",
    )

    # MLflow callback
    mlflow_callback = MLflowLoggerCallback(
        tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=experiment_name,
        save_artifact=True,
    )

    # Run config
    run_config = RunConfig(
        callbacks=[mlflow_callback],
        checkpoint_config=checkpoint_config,
    )

    # Dataset
    ds = data.load_data(
        num_samples=train_loop_config["num_samples"],
        num_partitions=num_cpu_workers,
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
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=train_loop_config,
        scaling_config=scaling_config,
        run_config=run_config,
        datasets={"train": train_ds, "val": val_ds},
        dataset_config=dataset_config,
        preprocessor=preprocessor,
    )

    # Train
    results = trainer.fit()
    d = {
        "timestamp": int(datetime.datetime.now().timestamp()),
        "run_id": utils.get_run_id(experiment_name=experiment_name, trial_id=results.metrics["trial_id"]),
        "params": results.config["train_loop_config"],
        "metrics": results.metrics_dataframe.to_dict(),
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
