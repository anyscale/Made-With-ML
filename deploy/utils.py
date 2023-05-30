import json
import subprocess
import tempfile

import boto3
import typer
import yaml
from anyscale import AnyscaleSDK

# Initialize
app = typer.Typer()


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
    project_id = last_line.split()[1]
    print(project_id)
    return project_id


@app.command()
def get_latest_cluster_env_build_id(
    cluster_env_name: str = typer.Option(..., "--cluster-env-name", "-n", help="name of the cluster environment")
) -> str:
    """Get the latest cluster environment build id."""
    sdk = AnyscaleSDK()
    res = sdk.search_cluster_environments({"name": {"equals": cluster_env_name}})
    apt_id = res.results[0].id
    res = sdk.list_cluster_environment_builds(apt_id)
    bld_id = res.results[-1].id
    print(bld_id)
    return bld_id


@app.command()
def submit_job(
    yaml_config_fp: str = typer.Option(..., "--yaml-config-fp", help="path of the job's yaml config file"),
    cluster_env_name: str = typer.Option(..., "--cluster-env-name", help="cluster environment's name"),
    run_id: str = typer.Option("", "--run-id", help="run ID to use to execute ML workflow"),
    username: str = typer.Option("", "--username", help="GitHub username"),
    commit_id: str = typer.Option("default", "--commit-id", help="used as UUID to store results to S3"),
) -> None:
    """Submit a job to Anyscale."""
    # Load yaml config
    with open(yaml_config_fp, "r") as file:
        yaml_config = yaml.safe_load(file)

    # Edit yaml config
    yaml_config["build_id"] = get_latest_cluster_env_build_id(cluster_env_name=cluster_env_name)
    yaml_config["runtime_env"]["env_vars"]["run_id"] = run_id
    yaml_config["runtime_env"]["env_vars"]["username"] = username
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


def to_markdown(data):
    """Convert a dict to markdown."""
    # Compose markdown
    markdown = ""
    for key, value in data.items():
        markdown += f"**{key}:**\n\n"

        if isinstance(value, dict):
            markdown += "| Key | Value |\n"
            markdown += "| --- | --- |\n"
            for nested_key, nested_value in value.items():
                markdown += f"| {nested_key} | {nested_value} |\n"

        elif isinstance(value, list) and all(isinstance(item, dict) for item in value):
            if value:
                headers = sorted(set().union(*[item.keys() for item in value]))
                markdown += "| " + " | ".join(headers) + " |\n"
                markdown += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                for item in value:
                    markdown += "| " + " | ".join([str(item.get(header, "")) for header in headers]) + " |\n"
            else:
                markdown += "(empty list)\n"

        else:
            markdown += f"{value}\n"

        markdown += "\n"
    return markdown


@app.command()
def json_to_markdown(
    json_fp: str = typer.Option(..., "--json-fp", "-fp", help="path of json file to convert to markdown"),
    markdown_fp: str = typer.Option(..., "--markdown-fp", help="path of markdown file to save to"),
):
    """Convert a json file to markdown."""
    # Read JSON file
    with open(json_fp, "r") as file:
        data = json.load(file)

    # Convert to markdown
    markdown = to_markdown(data)

    # Save to markdown file
    with open(markdown_fp, "w") as file:
        file.write(markdown)
    return markdown


if __name__ == "__main__":
    app()
