from pathlib import Path

import numpy as np
import pandas as pd
import clarite

TESTS_PATH = Path(__file__).parent.parent
DATA_PATH = TESTS_PATH / 'test_data_files'
RESULT_PATH = TESTS_PATH / 'r_test_output' / 'interactions'


def load_r_results(filename):
    """Load directly calculated results (from R) and convert column names to match python results"""
    r_result = pd.read_csv(filename)
    r_result = r_result.set_index('Variable')
    r_result[["Beta", "SE", "Diff_AIC", "pvalue", "N"]] = \
        r_result[["Beta", "SE", "Diff_AIC", "pvalue", "N"]].astype('float64')
    return r_result


def compare_result(loaded_result, python_result, atol=0, rtol=1e-04):
    """Compare loaded results (run directly using the survey lib) to CLARITE results, with optional tolerances"""
    # Remove "Phenotype" from the index in calculated results
    python_result.reset_index(drop=False).set_index('Variable').drop(columns=['Phenotype'])

    # Convert 'N' from IntegerArray to float
    python_result = python_result.astype({"N": 'float'})

    # Merge and ensure no rows are dropped
    loaded_result = loaded_result.add_suffix('_loaded')
    python_result = python_result.add_suffix('_python')
    merged = pd.merge(left=loaded_result, right=python_result, left_index=True, right_index=True, how="inner")
    try:
        assert len(merged) == len(loaded_result) == len(python_result)
    except AssertionError:
        raise ValueError(f" Loaded Results have {len(loaded_result):,} rows,"
                         f" Python results have {len(python_result):,} rows,"
                         f" merged data has {len(merged):,} rows")

    # Same variant_type
    try:
        assert merged["Variable_type_loaded"].equals(merged["Variable_type_python"])
    except AssertionError:
        raise ValueError(f"Variable_type:\n"
                         f"{merged[f'Variable_type_loaded']}\n"
                         f"{merged[f'Variable_type_python']}\n")
    # Close-enough equality of numeric values
    for var in ["N", "Beta", "SE", "pvalue"]:
        try:
            assert np.allclose(merged[f"{var}_loaded"], merged[f"{var}_python"],
                               equal_nan=True, atol=atol, rtol=rtol)
        except AssertionError:
            raise ValueError(f"{var}:\n"
                             f"{merged[f'{var}_loaded']}\n"
                             f"{merged[f'{var}_python']}\n")
    for var in ["Diff_AIC"]:
        try:
            # Value must be close when both exist or both are NaN
            assert np.allclose(merged[f"{var}_loaded"],
                               merged[f"{var}_python"], equal_nan=True)
        except AssertionError:
            raise ValueError(f"{var}: Loaded ({merged[f'{var}_loaded']}) != Python ({merged[f'{var}_python']})")


##################
# NHANES Dataset #
##################
# A data frame with 8591 observations on the following 7 variables.
# SDMVPSU - Primary sampling units
# SDMVSTRA - Sampling strata
# WTMEC2YR - Sampling weights
# HI_CHOL - Binary: 1 for total cholesterol over 240mg/dl, 0 under 240mg/dl
# race - Categorical (1=Hispanic, 2=non-Hispanic white, 3=non-Hispanic black, 4=other)
# agecat  - Categorical Age group(0,19] (19,39] (39,59] (59,Inf]
# RIAGENDR - Binary: Gender: 1=male, 2=female


def test_interactions_nhanes_ageXgender(data_NHANES):
    """Test the nhanes dataset a specific interaction"""
    # Process data
    df = clarite.modify.colfilter(data_NHANES, only=["HI_CHOL", "RIAGENDR", "race", "agecat"])
    # Get Results
    # loaded_result = load_r_results(RESULT_PATH / "nhanes_noweights_result.csv")
    python_result = clarite.analyze.interactions(outcome_variable="HI_CHOL",
                                                 covariates=["race"],
                                                 data=df,
                                                 interactions=[("agecat", "RIAGENDR")],
                                                 report_betas=True)
    python_result_nobeta = clarite.analyze.interactions(outcome_variable="HI_CHOL",
                                                        covariates=["race"],
                                                        data=df,
                                                        interactions=[("agecat", "RIAGENDR")],
                                                        report_betas=False)
    print()
    ## Compare
    # compare_result(loaded_result, python_result)


def test_interactions_nhanes_pairwise(data_NHANES):
    """Test the nhanes dataset with pairwise interactions of the three variables"""
    # Process data
    df = clarite.modify.colfilter(data_NHANES, only=["HI_CHOL", "RIAGENDR", "race", "agecat"])
    # Get Results
    # loaded_result = load_r_results(RESULT_PATH / "nhanes_noweights_result.csv")
    python_result = clarite.analyze.interactions(outcome_variable="HI_CHOL",
                                                 covariates=[],
                                                 data=df,
                                                 interactions=None,
                                                 report_betas=True)
    python_result_nobeta = clarite.analyze.interactions(outcome_variable="HI_CHOL",
                                                        covariates=[],
                                                        data=df,
                                                        interactions=None,
                                                        report_betas=False)
    clarite.analyze.add_corrected_pvalues(python_result, pvalue='Beta_pvalue')
    clarite.analyze.add_corrected_pvalues(python_result, pvalue='LRT_pvalue', groupby="Test_Number")
    # Compare
    # compare_result(loaded_result, python_result)
