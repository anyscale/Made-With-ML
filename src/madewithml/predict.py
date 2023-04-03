# madewithml/predict.py
import mlflow
from ray.train.torch import TorchCheckpoint

def predict_tag(texts, tokenizer, model, label_encoder):
    encoded_input = tokenizer(texts, return_tensors="pt", padding=True)
    inputs = [encoded_input["input_ids"], encoded_input["attention_mask"]]
    y_pred = model.predict(inputs=inputs)
    label = label_encoder.decode(y_pred)
    return label


def predict(texts, experiment_name=None):
    """Predict labels using a model from an experiment (or best experiment)."""
    # Sorted runs
    sorted_runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=["metrics.val_loss ASC"],
        search_all_experiments=True,  # only honored if experiment_names is None
    )

