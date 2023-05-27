import boto3
import typer
from anyscale import AnyscaleSDK
from anyscale.sdk.anyscale_client import models
import yaml
from pathlib import Path
import subprocess
import tempfile

# Initialize
app = typer.Typer()
sdk = AnyscaleSDK()


@app.command()
def get_project_id(project_name: str = "") -> str:
    """Get the project id."""
    output = subprocess.run(
        ["anyscale", "project", "list", "--name=" + project_name, "--created-by-me"], capture_output=True, text=True
    ).stdout
    lines = output.split("\n")
    matching_lines = [line for line in lines if project_name in line]
    last_line = matching_lines[-1] if matching_lines else None
    project_id = last_line.split()[0]
    return project_id


@app.command()
def get_latest_cluster_env_build_id(cluster_env_name: str = "") -> str:
    """Get the latest cluster environment build id."""
    res = sdk.search_cluster_environments({"name": {"equals": cluster_env_name}})
    apt_id = res.results[0].id
    res = sdk.list_cluster_environment_builds(apt_id)
    bld_id = res.results[-1].id
    return bld_id


@app.command()
def submit_job(yaml_config_fp: str = "", cluster_env_name: str = "", run_id: str = "", commit_id: str = "default") -> None:
    """Submit a job to Anyscale."""
    # Load yaml config
    with open(yaml_config_fp, "r") as file:
        yaml_config = yaml.safe_load(file)

    # Edit yaml config
    yaml_config["build_id"] = get_latest_cluster_env_build_id(cluster_env_name=cluster_env_name)
    yaml_config["runtime_env"]["env_vars"]["run_id"] = run_id
    yaml_config["runtime_env"]["env_vars"]["commit_id"] = commit_id

    # Execute Anyscale job
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=True, mode="w+b") as temp_file:
        temp_file_path = temp_file.name
        yaml.dump(yaml_config, temp_file, encoding="utf-8")
        subprocess.run(["anyscale", "job", "submit", "--wait", temp_file_path])


@app.command()
def save_to_s3(file: str = "", bucket_name: str = "", path: str = "") -> None:
    """Save file to S3 bucket."""
    s3 = boto3.client("s3")
    s3.upload_file(file, bucket_name, path)


if __name__ == "__main__":
    app()