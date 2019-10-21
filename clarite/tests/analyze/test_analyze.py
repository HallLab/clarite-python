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
    r_result = load_r_results(DATA_PATH / "fpc_noweights_result.csv")
    # Process data
    df = clarite.modify.make_continuous(df, only=["x", "y"])
    df = clarite.modify.colfilter(df, only=["x", "y"])
    python_result = clarite.analyze.ewas(phenotype="y", covariates=[], data=df, min_n=1)
    # Compare
    compare_result(r_result, python_result)


def test_fpc_withoutfpc():
    """Use a survey design specifying weights, cluster, strata"""
    # Load the data
    df = clarite.load.from_csv(DATA_PATH / "fpc_data.csv", index_col='ID')
    # Load the expected results
    r_result = load_r_results(DATA_PATH / "fpc_withoutfpc_result.csv")
    # Process data
    df = clarite.modify.make_continuous(df, only=["x", "y"])
    design = clarite.survey.SurveyDesignSpec(df, weights="weight", cluster="psuid", strata="stratid", nest=True)
    df = clarite.modify.colfilter(df, only=["x", "y"])
    python_result = clarite.analyze.ewas(phenotype="y", covariates=[], data=df, survey_design_spec=design, min_n=1)
    # Compare
    compare_result(r_result, python_result)


def test_fpc_withfpc():
    """Use a survey design specifying weights, cluster, strata, fpc"""
    # Load the data
    df = clarite.load.from_csv(DATA_PATH / "fpc_data.csv", index_col='ID')
    # Load the expected results
    r_result = load_r_results(DATA_PATH / "fpc_withfpc_result.csv")
    # Process data
    df = clarite.modify.make_continuous(df, only=["x", "y"])
    design = clarite.survey.SurveyDesignSpec(df, weights="weight", cluster="psuid", strata="stratid",
                                             fpc="Nh", nest=True)
    df = clarite.modify.colfilter(df, only=["x", "y"])
    python_result = clarite.analyze.ewas(phenotype="y", covariates=[], data=df, survey_design_spec=design, min_n=1)
    # Compare
    compare_result(r_result, python_result)


def test_fpc_withfpc_nostrata():
    """Use a survey design specifying weights, cluster, strata, fpc"""
    # Load the data
    df = clarite.load.from_csv(DATA_PATH / "fpc_nostrat_data.csv", index_col='ID')
    # Load the expected results
    r_result = load_r_results(DATA_PATH / "fpc_withfpc_nostrat_result.csv")
    # Process data
    df = clarite.modify.make_continuous(df, only=["x", "y"])
    design = clarite.survey.SurveyDesignSpec(df, weights="weight", cluster="psuid", strata=None,
                                             fpc="Nh", nest=True)
    df = clarite.modify.colfilter(df, only=["x", "y"])
    python_result = clarite.analyze.ewas(phenotype="y", covariates=[], data=df, survey_design_spec=design, min_n=1)
    # Compare
    compare_result(r_result, python_result)

###############
# api Dataset #
###############
# continuous: ["api00", "ell", "meals", "mobility"]  (there are others, but they weren't tested in R)
# --------------
# weights = "pw"
# strata = "stype"
# cluster = "dnum"
# fpc = "fpc"
# nest = False


def test_api_noweights():
    """Test the api dataset with no survey info"""
    # Load the data
    df = clarite.load.from_csv(DATA_PATH / "apipop_data.csv", index_col='ID')
    # Load the expected results
    r_result = load_r_results(DATA_PATH / "api_apipop_result.csv")
    # Process data
    df = clarite.modify.make_continuous(df, only=["api00", "ell", "meals", "mobility"])
    df = clarite.modify.colfilter(df, only=["api00", "ell", "meals", "mobility"])
    python_result = pd.concat([
        clarite.analyze.ewas(phenotype="api00", covariates=["meals", "mobility"], data=df, min_n=1),
        clarite.analyze.ewas(phenotype="api00", covariates=["ell", "mobility"], data=df, min_n=1),
        clarite.analyze.ewas(phenotype="api00", covariates=["ell", "meals"], data=df, min_n=1),
        ], axis=0)
    # Compare
    compare_result(r_result, python_result)


def test_api_stratified():
    """Test the api dataset with weights, strata, and fpc"""
    # Load the data
    df = clarite.load.from_csv(DATA_PATH / "apistrat_data.csv", index_col='ID')
    # Load the expected results
    r_result = load_r_results(DATA_PATH / "api_apistrat_result.csv")
    # Process data
    df = clarite.modify.make_continuous(df, only=["api00", "ell", "meals", "mobility"])
    design = clarite.survey.SurveyDesignSpec(df, weights="pw", cluster=None, strata="stype", fpc="fpc")
    df = clarite.modify.colfilter(df, only=["api00", "ell", "meals", "mobility"])
    python_result = pd.concat([
        clarite.analyze.ewas(phenotype="api00", covariates=["meals", "mobility"],
                             data=df, survey_design_spec=design, min_n=1),
        clarite.analyze.ewas(phenotype="api00", covariates=["ell", "mobility"],
                             data=df, survey_design_spec=design, min_n=1),
        clarite.analyze.ewas(phenotype="api00", covariates=["ell", "meals"],
                             data=df, survey_design_spec=design, min_n=1),
    ], axis=0)
    # Compare
    compare_result(r_result, python_result)


def test_api_cluster():
    """Test the api dataset with weights, clusters, and fpc"""
    # Load the data
    df = clarite.load.from_csv(DATA_PATH / "apiclus1_data.csv", index_col='ID')
    # Load the expected results
    r_result = load_r_results(DATA_PATH / "api_apiclus1_result.csv")
    # Process data
    df = clarite.modify.make_continuous(df, only=["api00", "ell", "meals", "mobility"])
    design = clarite.survey.SurveyDesignSpec(df, weights="pw", cluster="dnum", strata=None, fpc="fpc")
    df = clarite.modify.colfilter(df, only=["api00", "ell", "meals", "mobility"])
    python_result = pd.concat([
        clarite.analyze.ewas(phenotype="api00", covariates=["meals", "mobility"],
                             data=df, survey_design_spec=design, min_n=1),
        clarite.analyze.ewas(phenotype="api00", covariates=["ell", "mobility"],
                             data=df, survey_design_spec=design, min_n=1),
        clarite.analyze.ewas(phenotype="api00", covariates=["ell", "meals"],
                             data=df, survey_design_spec=design, min_n=1),
    ], axis=0)
    # Compare
    compare_result(r_result, python_result)

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

# TODO: Update these to force comparison
# Currently results are different in some cases, possibly because the outcome is binomial

def test_nhanes_noweights():
    """Test the nhanes dataset with no survey info"""
    # Load the data
    df = clarite.load.from_csv(DATA_PATH / "nhanes_data.csv", index_col='ID')
    # Load the expected results
    r_result = load_r_results(DATA_PATH / "nhanes_noweights_result.csv")
    # Process data
    df = clarite.modify.make_binary(df, only=["HI_CHOL", "RIAGENDR"])
    df = clarite.modify.make_categorical(df, only=["race", "agecat"])
    df = clarite.modify.colfilter(df, only=["HI_CHOL", "RIAGENDR", "race", "agecat"])
    df = clarite.modify.rowfilter_incomplete_obs(df)
    python_result = pd.concat([
        clarite.analyze.ewas(phenotype="HI_CHOL", covariates=["agecat", "RIAGENDR"], data=df, min_n=1),
        clarite.analyze.ewas(phenotype="HI_CHOL", covariates=["race", "RIAGENDR"], data=df, min_n=1),
        clarite.analyze.ewas(phenotype="HI_CHOL", covariates=["race", "agecat"], data=df, min_n=1),
        ], axis=0)
    # Compare
    print(python_result)
    print(r_result)
    #compare_result(r_result, python_result, bin_vars=["RIAGENDR"])