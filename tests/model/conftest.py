import pytest


def pytest_addoption(parser):
    parser.addoption("--run-id", action="store", default=None, help="Run ID of model to use.")


@pytest.fixture(scope="module")
def run_id(request):
    return request.config.getoption("--run-id")
