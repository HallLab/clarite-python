from pathlib import Path

import clarite
import pytest
import pandas_genomics as pg

TESTS_PATH = Path(__file__).parent.parent
DATA_PATH = TESTS_PATH / "test_data_files"


# Dataset fixtures
@pytest.fixture
def data_all_types():
    # Load nhanes data
    df = clarite.load.from_csv(DATA_PATH / "nhanes_data.csv", index_col=None)
    # Add some random genotypes
    var1 = pg.scalars.Variant(
        chromosome="1", position=123, id="rs123", ref="A", alt="T"
    )
    var2 = pg.scalars.Variant(
        chromosome="2", position=456, id="rs456", ref="G", alt=["T", "C"]
    )
    df["var1"] = pg.sim.generate_random_gt(var1, alt_allele_freq=0.5, n=len(df))
    df["var2"] = pg.sim.generate_random_gt(var2, alt_allele_freq=[0.2, 0.1], n=len(df))
    # Add unknown and constant
    df["unknown"] = ["unknown"] * len(df)
    df["constant"] = [1] * len(df)
    # Update types
    df = clarite.modify.make_binary(df, only=["HI_CHOL", "RIAGENDR"])
    df = clarite.modify.make_categorical(df, only=["race", "agecat", "constant"])
    return df
