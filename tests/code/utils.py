import uuid

import mlflow


def generate_experiment_name(prefix: str = "test") -> str:
    """Generate a unique experiment name.

    Args:
        prefix (str, optional): prefix to add to unique experiment name. Defaults to "test".

    Returns:
        str: unique experiment name.
    """
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def delete_experiment(experiment_name: str) -> None:
    """Delete an experiment.

    Args:
        experiment_name (str): name of experiment to delete.
    """
    client = mlflow.tracking.MlflowClient()
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
    client.delete_experiment(experiment_id=experiment_id)
