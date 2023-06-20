import sys

sys.path.append(".")

import mlflow  # NOQA: E402

from madewithml.config import MLFLOW_TRACKING_URI  # NOQA: E402
from madewithml.serve import ModelDeployment  # NOQA: E402

# Entrypoint
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
run_id = [line.strip() for line in open("/mnt/user_storage/run_id.txt")][0]
entrypoint = ModelDeployment.bind(run_id=run_id, threshold=0.9)
