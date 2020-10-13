from pathlib import Path

import clarite
import pytest

TESTS_PATH = Path(__file__).parent.parent
DATA_PATH = TESTS_PATH / 'test_data_files'


# Dataset fixtures
@pytest.fixture
def data_fpc():
    # Load the data
    df = clarite.load.from_csv(DATA_PATH / "fpc_data.csv", index_col=None)
    # Process data
    df = clarite.modify.make_continuous(df, only=["x", "y"])
    return df


@pytest.fixture()
def data_NHANES():
    df = clarite.load.from_csv(DATA_PATH / "nhanes_data.csv", index_col=None)
    df = clarite.modify.make_binary(df, only=["HI_CHOL", "RIAGENDR"])
    df = clarite.modify.make_categorical(df, only=["race", "agecat"])
    return df


@pytest.fixture()
def data_NHANES_withNA():
    df = clarite.load.from_csv(DATA_PATH / "nhanes_NAs_data.csv", index_col=None)
    df = clarite.modify.make_binary(df, only=["HI_CHOL", "RIAGENDR"])
    df = clarite.modify.make_categorical(df, only=["race", "agecat"])
    return df


@pytest.fixture()
def data_NHANES_lonely():
    df = clarite.load.from_csv(DATA_PATH / "nhanes_lonely_data.csv", index_col=None)
    df = clarite.modify.make_binary(df, only=["HI_CHOL", "RIAGENDR"])
    df = clarite.modify.make_categorical(df, only=["race", "agecat"])
    return df
