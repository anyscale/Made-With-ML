import os
import subprocess  # Workaround: Fetch from s3

import mlflow

from madewithml.config import MLFLOW_TRACKING_URI, MODEL_REGISTRY
from madewithml.predict import get_best_run_id
from madewithml.serve import CustomLogic, Model

BUCKET = os.environ.get("BUCKET", "s3://goku-mlops")
RUN_ID = os.environ.get("RUN_ID", None)

subprocess.check_output(["aws", "s3", "sync", f"${BUCKET}/mlflow", str(MODEL_REGISTRY)])
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
experiment_name = os.getenv("EXPERIMENT_NAME", "llm")

if RUN_ID:
    run_id = RUN_ID
else:
    run_id = get_best_run_id(experiment_name, "val_loss", "ASC")

entrypoint = CustomLogic.bind(model=Model.bind(run_id=run_id), threshold=0.9)
