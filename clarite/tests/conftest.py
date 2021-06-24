import pytest
from statsmodels import datasets


# Datasets for testing
@pytest.fixture
def plantTraits():
    data = datasets.get_rdataset("plantTraits", "cluster", cache=True).data
    data.index.name = "ID"
    return data
