import subprocess
import sys
import tempfile
from typing import Dict

import yaml


def substitute_env_vars(data: Dict, env_vars: Dict) -> Dict:
    for key, value in data.items():
        if key in env_vars:
            data[key] = env_vars[key]
        if isinstance(value, dict):
            data[key] = substitute_env_vars(value, env_vars)
    return data


def execute_job(file_path: str, env_vars: Dict) -> None:
    """Execute the Anyscale job with modified yaml config."""
    # Modify yaml with environment variables
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    data = substitute_env_vars(data, env_vars)

    # Execute Anyscale job
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=True, mode="w+b") as temp_file:
        temp_file_path = temp_file.name
        yaml.dump(data, temp_file, encoding="utf-8")
        subprocess.run(["anyscale", "job", "submit", "--wait", temp_file_path])


if __name__ == "__main__":
    file_path = sys.argv[1]
    env_vars = dict(arg.split("=") for arg in sys.argv[2:])
    execute_job(file_path, env_vars)
