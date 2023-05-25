import great_expectations as ge
import pandas as pd
import pytest


def pytest_addoption(parser):
    parser.addoption("--dataset-loc", action="store", default=None, help="Dataset location.")


@pytest.fixture(scope="module")
def df(request):
    dataset_loc = request.config.getoption("--dataset-loc")
    df = ge.dataset.PandasDataset(pd.read_csv(dataset_loc))
    return df
