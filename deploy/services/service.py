import os

import mlflow

from madewithml.config import MLFLOW_TRACKING_URI
from madewithml.serve import CustomLogic, Model

# Entrypoint
RUN_ID = os.getenv("RUN_ID")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
entrypoint = CustomLogic.bind(model=Model.bind(run_id=RUN_ID), threshold=0.9)
