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


def compare_result(r_result, python_result):
    merged = pd.merge(left=r_result, right=python_result,
                      left_index=True, right_index=True,
                      how="inner", suffixes=("_r", "_python"))
    try:
        assert len(merged) == len(r_result) == len(python_result)
    except AssertionError:
        raise ValueError(f"R Results have {len(r_result):,} rows,"
                         f" Python results have {len(python_result):,} rows,"
                         f" merged data has {len(merged):,} rows")
    # Close-enough equality of numeric values
    for var in ["N", "Beta", "SE", "Variable_pvalue", "LRT_pvalue", "Diff_AIC", "pvalue"]:
        try:
            assert np.allclose(merged[f"{var}_r"], merged[f"{var}_python"], equal_nan=True)
        except AssertionError:
            raise ValueError(f"{var}: R ({merged[f'{var}_r']}) != Python ({merged[f'{var}_python']})")
    # Both converged
    assert all(merged["Converged_r"] == merged["Converged_python"])


def test_fpc_noweights():
    # Load the data
    df = clarite.load.from_csv(DATA_PATH / "fpc_data.csv", index_col='ID')
    # Load the expected results
    r_result = load_r_results(DATA_PATH / "fpc_noweights_result.csv")
    # Process data
    df = clarite.modify.make_continuous(df, only=["x", "y"])
    df = df.drop(columns=["stratid", "psuid", "weight", "nh", "Nh"])
    python_result = clarite.analyze.ewas(phenotype="y", covariates=[], data=df, min_n=1)
    # Compare
    compare_result(r_result, python_result)
