import os
import sys

sys.path.append(".")

# Workaround: Fetch from s3
import subprocess  # NOQA: E402

import mlflow  # NOQA: E402

from madewithml.config import MLFLOW_TRACKING_URI, MODEL_REGISTRY  # NOQA: E402
from madewithml.predict import get_best_run_id  # NOQA: E402
from madewithml.serve import ModelDeployment  # NOQA: E402

s3_bucket = os.getenv("S3_BUCKET")
username = os.getenv("GITHUB_USERNAME")
subprocess.check_output(["aws", "s3", "sync", f"{s3_bucket}/{username}/mlflow", str(MODEL_REGISTRY)])

# Entrypoint
run_id = os.getenv("RUN_ID")
if not run_id:
    experiment_name = os.getenv("EXPERIMENT_NAME")
    run_id = get_best_run_id(experiment_name=experiment_name, metric="val_loss", mode="ASC")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
entrypoint = ModelDeployment.bind(run_id=run_id, threshold=0.9)
