import mlflow

from madewithml.config import MLFLOW_TRACKING_URI, MODEL_REGISTRY
from madewithml.predict import get_best_run_id
from madewithml.serve import CustomLogic, Model

# Workaround: Fetch from s3
import subprocess

subprocess.check_output(["aws", "s3", "sync", "s3://kf-mlops-dev/mlflow", str(MODEL_REGISTRY)])

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


best_run_id = get_best_run_id("llm", "val_loss", "ASC")
entrypoint = CustomLogic.bind(model=Model.bind(run_id=best_run_id), threshold=0.9)
