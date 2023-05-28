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
def get_project_id(
    project_name: str = typer.Option(..., "--project-name", "-n", help="name of the Anyscale project")
) -> str:
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
def get_latest_cluster_env_build_id(
    cluster_env_name: str = typer.Option(..., "--cluster-env-name", "-n", help="name of the cluster environment")
) -> str:
    """Get the latest cluster environment build id."""
    res = sdk.search_cluster_environments({"name": {"equals": cluster_env_name}})
    apt_id = res.results[0].id
    res = sdk.list_cluster_environment_builds(apt_id)
    bld_id = res.results[-1].id
    return bld_id


@app.command()
def submit_job(
    yaml_config_fp: str = typer.Option(..., "--yaml-config-fp", help="path of the job's yaml config file"),
    cluster_env_name: str = typer.Option(..., "--cluster-env-name", help="cluster environment's name"),
    run_id: str = typer.Option("", "--run-id", help="run ID to use to execute ML workflow"),
    commit_id: str = typer.Option("default", "--commit-id", help="used as UUID to store results to S3"),
) -> None:
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
def save_to_s3(
    file_path: str = typer.Option(..., "--file-path", "-fp", help="path of file to save to S3"),
    bucket_name: str = typer.Option(..., "--bucket-name", help="name of S3 bucket (without s3:// prefix)"),
    bucket_path: str = typer.Option(..., "--bucket-path", help="path in S3 bucket to save to"),
) -> None:
    """Save file to S3 bucket."""
    s3 = boto3.client("s3")
    s3.upload_file(file_path, bucket_name, bucket_path)


@app.command()
def add_setup_to_jobs(input_file_path: str = typer.Option(..., "--input-file-path", "-fp", help="template file"),
                      setup_steps_path: str = typer.Option(..., "--setup-steps-path", "-fp", help="steps to add"),
                      output_file_path: str = typer.Option(..., "--output-file-path", "-fp", help="workflow file")):
    """Insert content from YAML file B into YAML file A as the first N steps of every job in file A.
    If the first N steps of each job in file A are already the steps from file B, then no changes are made.
    This is a workaround so that we don't have to copy/paste setup code for each GitHub Action job."""
    with open(input_file_path, "r") as input_file, \
         open(setup_steps_path, "r") as setup_steps_file, \
         open(output_file_path, "w") as output_file:

        indentation = " " * 6
        additional_text = setup_steps_file.read()
        for line in input_file:
            line = line.rstrip("\n")  # Remove trailing newline character
            output_file.write(line + "\n")  # Write the line as-is

            if "steps:" in line:
                # Write the additional text with the appropriate indentation
                output_file.write("\n")
                for additional_line in additional_text.split("\n"):
                    output_file.write(indentation + additional_line + "\n")

if __name__ == "__main__":
    app()
