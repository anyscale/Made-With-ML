import typer
from anyscale import AnyscaleSDK
import boto3
import subprocess

# Initialize
app = typer.Typer()
sdk = AnyscaleSDK()


@app.command()
def get_project_id(project_name: str = "") -> str:
    """Get the project id.

    Args:
        name (str): project name.

    Returns:
        str: project id.
    """
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
    """Get the latest cluster environment build id.

    Args:
        name (str): cluster environment name.

    Returns:
        str: cluster environment build id.
    """
    res = sdk.search_cluster_environments({"name": {"equals": cluster_env_name}})
    apt_id = res.results[0].id
    res = sdk.list_cluster_environment_builds(apt_id)
    bld_id = res.results[-1].id
    print (bld_id)
    return bld_id


@app.command()
def save_to_s3(file: str = "", bucket: str = "", path: str = "") -> None:
    """Save file to S3 bucket.

    Args:
        file (str): file path.
        bucket (str): S3 bucket name.
        path (str): S3 path.
    """
    s3 = boto3.client("s3")
    bucket_name = bucket.replace("s3://", "")
    s3.upload_file(file, bucket_name, path)


if __name__ == "__main__":
    app()