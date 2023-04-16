# madewithml/train.py
import numpy as np
import ray
from ray.air import Checkpoint, session
from ray.air.config import CheckpointConfig, RunConfig, ScalingConfig
from ray.air.integrations.mlflow import MLflowLoggerCallback
import ray.train as train
from ray.train.torch import TorchTrainer
import torch
import torch.nn as nn
from transformers import BertModel
import typer

from config import config
from madewithml import data, models, utils

# Initialize Typer CLI app
app = typer.Typer()


def train_step(dataloader, model, loss_fn, optimizer):
    """Train step."""
    size = len(dataloader.dataset) // session.get_world_size()
    model.train()
    loss = 0.0
    for i, batch in enumerate(dataloader):
        # batch = [item.to(device) for item in batch]  # Ray takes care of this
        inputs, targets = batch[:-1], batch[-1]
        optimizer.zero_grad()  # Reset gradients
        z = model(inputs)  # Forward pass
        J = loss_fn(z, targets)  # Define loss
        J.backward()  # Backward pass
        optimizer.step()  # Update weights

        # Cumulative metrics
        loss += (J.detach().item() - loss) / (i + 1)

    return loss


def eval_step(dataloader, model, loss_fn):
    """Eval step."""
    size = len(dataloader.dataset) // session.get_world_size()
    model.eval()
    loss = 0.0
    y_trues, y_preds = [], []
    with torch.inference_mode():
        for i, batch in enumerate(dataloader):
            # batch = [item.to(device) for item in batch]  # Ray takes care of this
            inputs, y_true = batch[:-1], batch[-1]
            z = model(inputs)  # Forward pass
            J = loss_fn(z, y_true).item()

            # Cumulative Metrics
            loss += (J - loss) / (i + 1)

            # Store outputs
            y_trues.extend(torch.argmax(y_true, dim=1).cpu().numpy())
            y_preds.extend(torch.argmax(z, dim=1).cpu().numpy())  # F.softmax(z).cpu().numpy()

    return loss, np.vstack(y_trues), np.vstack(y_preds)


# Training loop
def train_loop_per_worker(args):

    # Hyperparameters
    dropout_p = args["dropout_p"]
    lr = args["lr"]
    lr_factor = args["lr_factor"]
    lr_patience = args["lr_patience"]
    batch_size = args["batch_size"]
    num_epochs = args["num_epochs"]

    # Load data
    batch_size_per_worker = batch_size // session.get_world_size()
    train_data, val_data, test_data, label_encoder, class_weights = data.prep_data(args)
    train_dataloader, val_dataloader, test_dataloader = data.prep_data_loaders(
        train_data, val_data, test_data, batch_size=batch_size_per_worker)

    # Model
    llm = BertModel.from_pretrained("allenai/scibert_scivocab_uncased", return_dict=False)
    model = models.FinetunedLLM(
        llm=llm, dropout_p=dropout_p,
        embedding_dim=llm.config.hidden_size,
        num_classes=len(class_weights))
    model = train.torch.prepare_model(model)

    # Training components
    class_weights_tensor = torch.Tensor(np.array(list(class_weights.values()))).to(args["device"])
    loss_fn = nn.BCEWithLogitsLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=lr_patience)

    # Train
    best_val_loss = np.inf
    for epoch in range(num_epochs):
        train_loss = train_step(train_dataloader, model, loss_fn, optimizer)
        val_loss, _, _ = eval_step(val_dataloader, model, loss_fn)
        scheduler.step(val_loss)

        # Checkpoint
        artifacts = dict(epoch=epoch, model=model.state_dict(), class_to_index=label_encoder.class_to_index)
        metrics = dict(epoch=epoch, lr=optimizer.param_groups[0]['lr'], train_loss=train_loss, val_loss=val_loss)
        session.report(metrics, checkpoint=Checkpoint.from_dict(artifacts))


@app.command()
def train_model(experiment_name: str, num_workers: int = 1,
                use_gpu: bool = False, args_fp:str = config.ARGS_FP):
    """Main train function to train a model.
    Args:
        experiment_name (str): name of experiment.
        num_workers (int): number of workers to use for training.
        args_fp (str): location of args.
    """
    # Initialize Ray
    if ray.is_initialized():
        ray.shutdown()
    ray.init()

    # Checkpoint config
    checkpoint_config = CheckpointConfig(
        num_to_keep=1,  # num checkpoints to keep
        checkpoint_score_attribute="val_loss",
        checkpoint_score_order="min",
    )

    # Scaling config
    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=use_gpu,
        _max_cpu_fraction_per_node=0.8,
    )

    # Run config
    run_config = RunConfig(
        callbacks=[MLflowLoggerCallback(
            tracking_uri=config.MLFLOW_TRACKING_URI,
            experiment_name=experiment_name,
            save_artifact=True)],
        checkpoint_config=checkpoint_config,
    )

    # Trainer
    args = utils.load_dict(path=args_fp)
    args["device"] = "cuda" if use_gpu else "cpu"
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=args,
        scaling_config=scaling_config,
        run_config=run_config
    )

    # Train
    result = trainer.fit()
    return result


if __name__ == "__main__":
    app()