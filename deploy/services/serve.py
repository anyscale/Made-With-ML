import os
import sys

sys.path.append(".")

import mlflow  # NOQA: E402

from madewithml.config import MLFLOW_TRACKING_URI  # NOQA: E402
from madewithml.predict import get_best_run_id  # NOQA: E402
from madewithml.serve import CustomLogic, Model  # NOQA: E402

# Entrypoint
run_id = os.getenv("RUN_ID")
if not run_id:
    experiment_name = os.getenv("EXPERIMENT_NAME")
    run_id = get_best_run_id(experiment_name=experiment_name, metric="val_loss", mode="ASC")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
entrypoint = CustomLogic.bind(model=Model.bind(run_id=run_id), threshold=0.9)
