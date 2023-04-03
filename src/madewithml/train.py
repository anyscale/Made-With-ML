# madewithml/train.py
import numpy as np
from ray.air import Checkpoint, session
from ray.air.config import CheckpointConfig, RunConfig, ScalingConfig
from ray.air.integrations.mlflow import MLflowLoggerCallback
import ray.train as train
from ray.train.torch import TorchTrainer
import torch
import torch.nn as nn
from transformers import BertModel

from config import config
from madewithml import data, models, utils


def train_step(dataloader, model, loss_fn, optimizer):
    """Train step."""
    model.train()
    loss = 0.0
    for i, batch in enumerate(dataloader):
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
    model.eval()
    loss = 0.0
    y_trues, y_preds = [], []
    with torch.inference_mode():
        for i, batch in enumerate(dataloader):
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
def training_loop(args):
    """Training loop to train a model."""
    # Load data
    train_data, val_data, test_data, class_weights = data.prep_data(args)
    train_dataloader, val_dataloader, test_dataloader = data.prep_data_loaders(
        train_data, val_data, test_data, batch_size=args["batch_size"])

    # Model
    llm = BertModel.from_pretrained("allenai/scibert_scivocab_uncased", return_dict=False)
    model = models.FinetunedLLM(
        llm=llm, dropout_p=args["dropout_p"],
        embedding_dim=llm.config.hidden_size,
        num_classes=len(class_weights))
    model = train.torch.prepare_model(model)

    # Training components
    class_weights_tensor = torch.Tensor(np.array(list(class_weights.values())))
    loss_fn = nn.BCEWithLogitsLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5)

    # Train
    best_val_loss = np.inf
    for epoch in range(1, args["num_epochs"]+1):
        train_loss = train_step(train_dataloader, model, loss_fn, optimizer)
        val_loss, _, _ = eval_step(val_dataloader, model, loss_fn)
        scheduler.step(val_loss)
        checkpoint = Checkpoint.from_dict(dict(epoch=epoch, model=model.state_dict()))
        session.report({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}, checkpoint=checkpoint)


def train_model(experiment_name, num_workers, use_gpu, args_fp=config.ARGS_FP):
    """Main train function to train a model."""
    # Checkpoint config
    checkpoint_config = CheckpointConfig(
        num_to_keep=1,  # num checkpoints to keep
        checkpoint_score_attribute="val_loss",
        checkpoint_score_order="min"
        )

    # Run configuration
    run_config = RunConfig(
        callbacks=[MLflowLoggerCallback(
            tracking_uri=config.MLFLOW_TRACKING_URI,
            experiment_name=experiment_name,
            save_artifact=True)],
        checkpoint_config=checkpoint_config,
    )

    # Trainer
    args = utils.load_dict(path=args_fp)
    trainer = TorchTrainer(
        train_loop_per_worker=training_loop,
        train_loop_config=args,
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
        run_config=run_config
    )

    # Train
    result = trainer.fit()
    return result