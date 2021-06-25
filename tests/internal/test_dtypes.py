import pandas as pd
from pandas._testing import assert_series_equal

import clarite.internal.utilities as util


def test_get_dtype(data_all_types):
    assert util._get_dtype(data_all_types["SDMVPSU"]) == "continuous"
    assert util._get_dtype(data_all_types["HI_CHOL"]) == "binary"
    assert util._get_dtype(data_all_types["race"]) == "categorical"
    assert util._get_dtype(data_all_types["var1"]) == "genotypes"
    assert util._get_dtype(data_all_types["unknown"]) == "unknown"
    assert util._get_dtype(data_all_types["constant"]) == "constant"


def test_get_dtypes(data_all_types):
    clarite_dtypes = util._get_dtypes(data_all_types)
    assert_series_equal(
        clarite_dtypes,
        pd.Series(
            {
                "SDMVPSU": "continuous",
                "SDMVSTRA": "continuous",
                "WTMEC2YR": "continuous",
                "HI_CHOL": "binary",
                "race": "categorical",
                "agecat": "categorical",
                "RIAGENDR": "binary",
                "var1": "genotypes",
                "var2": "genotypes",
                "unknown": "unknown",
                "constant": "constant",
            }
        ),
    )
