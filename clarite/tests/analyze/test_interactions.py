from pathlib import Path

import numpy as np
import pandas as pd
import clarite

TESTS_PATH = Path(__file__).parent.parent
DATA_PATH = TESTS_PATH / "test_data_files"
RESULT_PATH = TESTS_PATH / "r_test_output" / "interactions"


def load_r_interaction_results(filename):
    """Load directly calculated results (from R) and convert column names to match python results"""
    r_result = pd.read_csv(filename, index_col=0)
    r_result[["Beta", "SE", "Beta_pvalue", "LRT_pvalue", "N"]] = r_result[
        ["Beta", "SE", "Beta_pvalue", "LRT_pvalue", "N"]
    ].astype("float64")
    r_result.index.name = "Interaction"
    return r_result


def compare_result(loaded_result, python_result, atol=0, rtol=1e-04):
    """Compare loaded results to CLARITE results, with optional tolerances"""
    # Remove "Phenotype" from the index in calculated results
    python_result = (
        python_result.reset_index(drop=False)
        .set_index("Interaction")
        .drop(columns=["Outcome"])
    )

    # Convert 'N' from IntegerArray to float
    python_result = python_result.astype({"N": "float"})

    # Merge and ensure no rows are dropped
    loaded_result = loaded_result.add_suffix("_loaded")
    python_result = python_result.add_suffix("_python")
    merged = pd.merge(
        left=loaded_result,
        right=python_result,
        left_index=True,
        right_index=True,
        how="inner",
    )
    try:
        assert len(merged) == len(loaded_result) == len(python_result)
    except AssertionError:
        raise ValueError(
            f" Loaded Results have {len(loaded_result):,} rows,"
            f" Python results have {len(python_result):,} rows,"
            f" merged data has {len(merged):,} rows"
        )

    # Close-enough equality of numeric values
    for var in ["N", "Beta", "SE", "Beta_pvalue", "LRT_pvalue"]:
        try:
            assert np.allclose(
                merged[f"{var}_loaded"],
                merged[f"{var}_python"],
                equal_nan=True,
                atol=atol,
                rtol=rtol,
            )
        except AssertionError:
            raise ValueError(
                f"{var}:\n"
                f"{merged[f'{var}_loaded']}\n"
                f"{merged[f'{var}_python']}\n"
            )


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
    df = clarite.modify.colfilter(
        data_NHANES, only=["HI_CHOL", "RIAGENDR", "race", "agecat"]
    )

    # No Betas
    loaded_result = load_r_interaction_results(RESULT_PATH / "nhanes_ageXgender.csv")
    python_result = clarite.analyze.interactions(
        outcome_variable="HI_CHOL",
        covariates=["race"],
        data=df,
        interactions=[("agecat", "RIAGENDR")],
        report_betas=False,
    )
    compare_result(loaded_result, python_result)

    # Betas - just test running since rows reported by R are different
    python_result = clarite.analyze.interactions(
        outcome_variable="HI_CHOL",
        covariates=["race"],
        data=df,
        interactions=[("agecat", "RIAGENDR")],
        report_betas=True,
    )

    # Two categoricals: Report one row for each combination, minus one (reference cat X reference cat = no beta value)
    assert (
        len(python_result)
        == (len(df["agecat"].cat.categories) * len(df["RIAGENDR"].cat.categories)) - 1
    )


def test_interactions_nhanes_weightXrace(data_NHANES):
    """
    Test the nhanes dataset a specific interaction (using 'weight' as a continuous variable).
    Not realistic, but good enough for a test.
    """
    # Process data
    df = clarite.modify.colfilter(
        data_NHANES, only=["HI_CHOL", "RIAGENDR", "race", "agecat", "WTMEC2YR"]
    )

    # No Betas
    loaded_result = load_r_interaction_results(RESULT_PATH / "nhanes_weightXrace.csv")
    python_result = clarite.analyze.interactions(
        outcome_variable="HI_CHOL",
        covariates=["agecat", "RIAGENDR"],
        data=df,
        interactions=[("WTMEC2YR", "race")],
        report_betas=False,
    )
    compare_result(loaded_result, python_result)

    # Betas
    loaded_result = load_r_interaction_results(RESULT_PATH / "nhanes_weightXrace_withbetas.csv")
    python_result = clarite.analyze.interactions(
        outcome_variable="HI_CHOL",
        covariates=["agecat", "RIAGENDR"],
        data=df,
        interactions=[("WTMEC2YR", "race")],
        report_betas=True,
    )
    # Remove brackets from index value to make comparison work
    python_result = python_result.reset_index(drop=False)
    python_result["Interaction"] = python_result['Interaction'].apply(
        lambda s: s.replace("[", "").replace("]", "")
    )
    python_result = python_result.set_index(["Interaction", "Outcome"])
    # Compare to R results
    compare_result(loaded_result, python_result)


def test_interactions_nhanes_pairwise(data_NHANES):
    """Test the nhanes dataset with pairwise interactions of the three variables"""
    # Process data
    df = clarite.modify.colfilter(
        data_NHANES, only=["HI_CHOL", "RIAGENDR", "race", "agecat"]
    )

    # No Betas
    loaded_result = load_r_interaction_results(RESULT_PATH / "nhanes_pairwise.csv")
    python_result = clarite.analyze.interactions(
        outcome_variable="HI_CHOL",
        covariates=[],
        data=df,
        interactions=None,
        report_betas=False,
    )
    compare_result(loaded_result, python_result)

    # Betas - just test running since rows reported by R are different
    python_result = clarite.analyze.interactions(
        outcome_variable="HI_CHOL",
        covariates=[],
        data=df,
        interactions=None,
        report_betas=True,
    )

    # Adding pvalues
    clarite.analyze.add_corrected_pvalues(python_result, pvalue="Beta_pvalue")
    clarite.analyze.add_corrected_pvalues(
        python_result, pvalue="LRT_pvalue", groupby="Test_Number"
    )
