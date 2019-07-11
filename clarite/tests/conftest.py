import pytest
import statsmodels as sm

# Datasets for testing
@pytest.fixture
def plantTraits():
    data = sm.datasets.get_rdataset('plantTraits', 'cluster', cache=True).data
    data.index.name = "ID"
    return data
