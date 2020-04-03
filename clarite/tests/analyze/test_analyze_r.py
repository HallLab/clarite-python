from pathlib import Path

import numpy as np
import pandas as pd
import clarite

DATA_PATH = Path(__file__).parent.parent / 'r_test_output'


def load_r_results(filename):
    """Load R results and convert column names to match python results"""
    r_result = pd.read_csv(filename)
    r_result = r_result.rename(columns={'pval': 'pvalue', 'phenotype': 'Phenotype'})
    r_result = r_result.set_index(['Variable', 'Phenotype'])
    return r_result


def compare_result(loaded_result, calculated_result):
    """Binary variables must be specified, since there are expected differences"""
    merged = pd.merge(left=loaded_result, right=calculated_result,
                      left_index=True, right_index=True,
                      how="inner", suffixes=("_loaded", "_calculated"))
    try:
        assert len(merged) == len(loaded_result) == len(calculated_result)
    except AssertionError:
        raise ValueError(f" Loaded Results have {len(loaded_result):,} rows,"
                         f" Calculated results have {len(calculated_result):,} rows,"
                         f" merged data has {len(merged):,} rows")
    # Close-enough equality of numeric values
    for var in ["N", "Beta", "SE", "Variable_pvalue", "LRT_pvalue", "pvalue"]:
        try:
            assert np.allclose(merged[f"{var}_loaded"], merged[f"{var}_calculated"], equal_nan=True, atol=0)
        except AssertionError:
            raise ValueError(f"{var}:\n"
                             f"{merged[f'{var}_loaded']}\n"
                             f"{merged[f'{var}_calculated']}")
    for var in ["Diff_AIC"]:
        # Pass if R result is NaN (quasibinomial) or Python result is NaN (survey data used)
        either_nan = merged[[f'{var}_loaded', f'{var}_calculated']].isna().any(axis=1)
        try:
            # Value must be close when both exist or both are NaN
            assert np.allclose(merged.loc[~either_nan, f"{var}_loaded"],
                               merged.loc[~either_nan, f"{var}_calculated"], equal_nan=True)
        except AssertionError:
            raise ValueError(f"{var}: Loaded ({merged[f'{var}_r']}) != Calculated ({merged[f'{var}_python']})")

    # Both converged
    assert all(merged["Converged_loaded"] == merged["Converged_calculated"])


###############
# fpc Dataset #
###############
# continuous: ["x", "y"]
# --------------
# weights = "weight"
# strata = "stratid"
# cluster = "psuid"
# fpc = "Nh"
# nest = True

def test_fpc_noweights():
    """Test the fpc dataset with no survey info"""
    # Load the data
    df = clarite.load.from_csv(DATA_PATH / "fpc_data.csv", index_col='ID')
    # Load the expected results
    loaded_result = load_r_results(DATA_PATH / "fpc_noweights_result.csv")
    # Process data
    df = clarite.modify.make_continuous(df, only=["x", "y"])
    df = clarite.modify.colfilter(df, only=["x", "y"])
    calculated_result = clarite.analyze.ewas_r(phenotype="y", covariates=[], data=df, min_n=1)
    # Compare
    compare_result(loaded_result, calculated_result)
