import pytest
import statsmodels as sm

# Datasets for testing
@pytest.fixture
def plantTraits():
    return sm.datasets.get_rdataset('plantTraits', 'cluster').data
