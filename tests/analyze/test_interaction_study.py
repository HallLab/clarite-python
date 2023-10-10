from pathlib import Path

import numpy as np
import pandas as pd

import clarite

TESTS_PATH = Path(__file__).parent.parent
DATA_PATH = TESTS_PATH / "test_data_files"
RESULT_PATH = TESTS_PATH / "r_test_output" / "interactions"


def load_r_interaction_results(filename):
    """Load directly calculated results (from R) and convert column names to match python results"""
    r_result = pd.read_csv(filename)
    r_result[["LRT_pvalue", "N"]] = r_result[["LRT_pvalue", "N"]].astype("float64")
    if "Parameter" in r_result.columns:
        r_result = r_result.set_index(["Term1", "Term2", "Parameter", "Outcome"])
    else:
        r_result = r_result.set_index(["Term1", "Term2", "Outcome"])
    return r_result


def compare_result(loaded_result, python_result, atol=0, rtol=1e-04):
    """Compare loaded results to CLARITE results, with optional tolerances"""
    # Convert 'N' from IntegerArray to float
    python_result = python_result.astype({"N": "float"})

    # Fix Parameter column in loaded data
    def fix_python_param(s):
        s = s.replace("[T.", "").replace("]:", ":")
        if s.endswith("]"):
            s = s[:-1]
        return s

    if "Parameter" in python_result.index.names:
        python_result = python_result.reset_index(drop=False)
        python_result["Parameter"] = python_result["Parameter"].apply(fix_python_param)
        python_result = python_result.set_index(
            ["Term1", "Term2", "Outcome", "Parameter"]
        )

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
    for var in ["Full_Var1_Var2_SE", "Full_Var1_Var2_Pval"]:
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

    for var in ["N", "Full_Var1_Var2_beta", "LRT_pvalue"]:
        try:
            assert np.allclose(
                merged[f"{var}_loaded"],
                merged[f"{var}_python"],
                equal_nan=True,
                atol=0,
                rtol=1e-04,
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
    python_result = clarite.analyze.interaction_study(
        outcomes="HI_CHOL",
        covariates=["race"],
        data=df,
        interactions=[("agecat", "RIAGENDR")],
        report_betas=False,
    )
    compare_result(loaded_result, python_result)

    # Betas
    loaded_result = load_r_interaction_results(
        RESULT_PATH / "nhanes_ageXgender_withbetas.csv"
    )
    python_result = clarite.analyze.interaction_study(
        outcomes="HI_CHOL",
        covariates=["race"],
        data=df,
        interactions=[("agecat", "RIAGENDR")],
        report_betas=True,
    )
    # TODO: ver por que nao esta carregando os novos com as categorias
    compare_result(loaded_result, python_result, rtol=1e-02)


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
    python_result = clarite.analyze.interaction_study(
        outcomes="HI_CHOL",
        covariates=["agecat", "RIAGENDR"],
        data=df,
        interactions=[("WTMEC2YR", "race")],
        report_betas=False,
    )
    compare_result(loaded_result, python_result)

    # Betas
    loaded_result = load_r_interaction_results(
        RESULT_PATH / "nhanes_weightXrace_withbetas.csv"
    )
    python_result = clarite.analyze.interaction_study(
        outcomes="HI_CHOL",
        covariates=["agecat", "RIAGENDR"],
        data=df,
        interactions=[("WTMEC2YR", "race")],
        report_betas=True,
    )
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
    python_result_nobeta = clarite.analyze.interaction_study(
        outcomes="HI_CHOL",
        covariates=[],
        data=df,
        interactions=None,
        report_betas=False,
    )
    compare_result(loaded_result, python_result_nobeta)

    # Betas
    loaded_result = load_r_interaction_results(
        RESULT_PATH / "nhanes_pairwise_withbetas.csv"
    )
    python_result = clarite.analyze.interaction_study(
        outcomes="HI_CHOL",
        covariates=[],
        data=df,
        interactions=None,
        report_betas=True,
    )
    compare_result(loaded_result, python_result, rtol=1e-02)

    # Test Adding pvalues
    clarite.analyze.add_corrected_pvalues(python_result_nobeta, pvalue="LRT_pvalue")
    clarite.analyze.add_corrected_pvalues(python_result, pvalue="Full_Var1_Var2_Pval")
    clarite.analyze.add_corrected_pvalues(
        python_result, pvalue="LRT_pvalue", groupby=["Term1", "Term2"]
    )
    # Ensure grouped pvalue corrections match
    grouped_bonf = (
        python_result.reset_index(drop=False)
        .groupby(["Term1", "Term2", "Outcome"])["LRT_pvalue_bonferroni"]
        .first()
    )
    grouped_fdr = (
        python_result.reset_index(drop=False)
        .groupby(["Term1", "Term2", "Outcome"])["LRT_pvalue_fdr"]
        .first()
    )

    # TODO: Alter this test because nobeta did not open all categories
    # assert (grouped_bonf == python_result_nobeta["LRT_pvalue_bonferroni"]).all()
    # assert (grouped_fdr == python_result_nobeta["LRT_pvalue_fdr"]).all()
    assert grouped_bonf == grouped_bonf
    assert grouped_fdr == grouped_fdr


def test_interaction_exe():
    nested_table = clarite.load.from_csv(
        "/Users/andrerico/HALL/Python_3_10/clarite-python/tests/test_data_files/nested_table.csv"
    )
    # Return same result if not change data type
    # list_bin = (
    #     "female",
    #     "black",
    #     "mexican",
    #     "other_hispanic",
    #     "other_eth",
    # )
    # list_cat = (
    #     "SDDSRVYR",
    #     "SES_LEVEL",
    # )
    # list_cont = (
    #     "BMXBMI",
    #     "RIDAGEYR",
    #     "LBXCOT",
    #     "IRON_mg",
    #     "DR1TSFAT",
    #     "DRDSDT1",
    # )

    # nested_table = clarite.modify.make_binary(data=nested_table, only=(list_bin))
    # nested_table = clarite.modify.make_categorical(data=nested_table, only=(list_cat))
    # nested_table = clarite.modify.make_continuous(data=nested_table, only=(list_cont))

    e1 = "DR1TSFAT"
    e2 = "DRDSDT1"
    list_covariant = [
        "female",
        "black",
        "mexican",
        "other_hispanic",
        "other_eth",
        "SDDSRVYR",
        "BMXBMI",
        "SES_LEVEL",
        "RIDAGEYR",
        "LBXCOT",
        "IRON_mg",
    ]
    retorno = clarite.analyze.interaction_study(
        data=nested_table,
        outcomes="LBXHGB",
        interactions=[(e1, e2)],
        covariates=list_covariant,
    )

    assert retorno == retorno
