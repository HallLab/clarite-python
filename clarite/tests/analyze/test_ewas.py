from pathlib import Path

import numpy as np
import pandas as pd

import clarite

TESTS_PATH = Path(__file__).parent.parent
DATA_PATH = TESTS_PATH / "test_data_files"
RESULT_PATH = TESTS_PATH / "r_test_output" / "analyze"


def load_surveylib_results(filename):
    """Load directly calculated results (from R) and convert column names to match python results"""
    r_result = pd.read_csv(filename)
    r_result = r_result.set_index("Variable")
    r_result[["Beta", "SE", "Diff_AIC", "pvalue", "N"]] = r_result[
        ["Beta", "SE", "Diff_AIC", "pvalue", "N"]
    ].astype("float64")
    return r_result


def compare_result(loaded_result, python_result, r_result, atol=0, rtol=1e-04):
    """Compare loaded results (run directly using the survey lib) to CLARITE results, with optional tolerances"""
    # Convert 'N' from IntegerArray to float
    python_result = python_result.astype({"N": "float"})
    r_result = r_result.astype({"N": "float"})

    # Merge and ensure no rows are dropped
    loaded_result = loaded_result.add_suffix("_loaded")
    python_result = python_result.add_suffix("_python")
    r_result = r_result.add_suffix("_r")
    merged = pd.merge(
        left=loaded_result,
        right=python_result,
        left_index=True,
        right_index=True,
        how="inner",
    )
    merged = pd.merge(
        left=merged, right=r_result, left_index=True, right_index=True, how="inner"
    )
    try:
        assert len(merged) == len(loaded_result) == len(python_result) == len(r_result)
    except AssertionError:
        raise ValueError(
            f" Loaded Results have {len(loaded_result):,} rows,"
            f" Python results have {len(python_result):,} rows,"
            f" R results have {len(r_result):,} rows,"
            f" merged data has {len(merged):,} rows"
        )

    # Same variant_type
    try:
        assert merged["Variable_type_loaded"].equals(merged["Variable_type_python"])
        assert merged["Variable_type_loaded"].equals(merged["Variable_type_r"])
    except AssertionError:
        raise ValueError(
            f"Variable_type:\n"
            f"{merged[f'Variable_type_loaded']}\n"
            f"{merged[f'Variable_type_python']}\n"
            f"{merged[f'Variable_type_r']}"
        )
    # Close-enough equality of numeric values
    for var in ["N", "Beta", "SE", "pvalue"]:
        try:
            assert np.allclose(
                merged[f"{var}_loaded"],
                merged[f"{var}_python"],
                equal_nan=True,
                atol=atol,
                rtol=rtol,
            )
            assert np.allclose(
                merged[f"{var}_python"],
                merged[f"{var}_r"],
                equal_nan=True,
                atol=atol,
                rtol=rtol,
            )
        except AssertionError:
            raise ValueError(
                f"{var}:\n"
                f"{merged[f'{var}_loaded']}\n"
                f"{merged[f'{var}_python']}\n"
                f"{merged[f'{var}_r']}"
            )
    for var in ["Diff_AIC"]:
        # Pass if loaded result is NaN (quasibinomial) or calculated result is NaN (survey data used)
        either_nan = merged[[f"{var}_loaded", f"{var}_python"]].isna().any(axis=1)
        try:
            # Value must be close when both exist or both are NaN
            assert np.allclose(
                merged.loc[~either_nan, f"{var}_loaded"],
                merged.loc[~either_nan, f"{var}_python"],
                equal_nan=True,
            )
        except AssertionError:
            raise ValueError(
                f"{var}: Loaded ({merged[f'{var}_loaded']}) != Python ({merged[f'{var}_python']})"
            )
        try:
            # Value must be close when both exist or both are NaN
            assert np.allclose(
                merged.loc[~either_nan, f"{var}_python"],
                merged.loc[~either_nan, f"{var}_r"],
                equal_nan=True,
            )
        except AssertionError:
            raise ValueError(
                f"{var}: Loaded ({merged[f'{var}_loaded']}) != R ({merged[f'{var}_r']})"
            )


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


def test_fpc_withoutfpc(data_fpc):
    """Use a survey design specifying weights, cluster, strata"""
    # Make Design
    design = clarite.survey.SurveyDesignSpec(
        data_fpc, weights="weight", cluster="psuid", strata="stratid", nest=True
    )
    df = clarite.modify.colfilter(data_fpc, only=["x", "y"])
    # Get results
    loaded_result = load_surveylib_results(RESULT_PATH / "fpc_withoutfpc_result.csv")
    python_result = clarite.analyze.ewas(
        outcome="y", covariates=[], data=df, survey_design_spec=design, min_n=1
    )
    r_result = clarite.analyze.ewas(
        outcome="y",
        covariates=[],
        data=df,
        survey_design_spec=design,
        min_n=1,
        regression_kind="r_survey",
    )
    # Compare
    compare_result(loaded_result, python_result, r_result)


def test_fpc_withfpc(data_fpc):
    """Use a survey design specifying weights, cluster, strata, fpc"""
    # Make Design
    design = clarite.survey.SurveyDesignSpec(
        data_fpc,
        weights="weight",
        cluster="psuid",
        strata="stratid",
        fpc="Nh",
        nest=True,
    )
    df = clarite.modify.colfilter(data_fpc, only=["x", "y"])
    # Get results
    loaded_result = load_surveylib_results(RESULT_PATH / "fpc_withfpc_result.csv")
    python_result = clarite.analyze.ewas(
        outcome="y", covariates=[], data=df, survey_design_spec=design, min_n=1
    )
    r_result = clarite.analyze.ewas(
        outcome="y",
        covariates=[],
        data=df,
        survey_design_spec=design,
        min_n=1,
        regression_kind="r_survey",
    )
    # Compare
    compare_result(loaded_result, python_result, r_result)


def test_fpc_withfpc_nostrata():
    """Use a survey design specifying weights, cluster, strata, fpc"""
    # Load the data
    df = clarite.load.from_csv(DATA_PATH / "fpc_nostrat_data.csv", index_col=None)
    # Process data
    df = clarite.modify.make_continuous(df, only=["x", "y"])
    design = clarite.survey.SurveyDesignSpec(
        df, weights="weight", cluster="psuid", strata=None, fpc="Nh", nest=True
    )
    df = clarite.modify.colfilter(df, only=["x", "y"])
    # Get results
    loaded_result = load_surveylib_results(
        RESULT_PATH / "fpc_withfpc_nostrat_result.csv"
    )
    python_result = clarite.analyze.ewas(
        outcome="y", covariates=[], data=df, survey_design_spec=design, min_n=1
    )
    r_result = clarite.analyze.ewas(
        outcome="y",
        covariates=[],
        data=df,
        survey_design_spec=design,
        min_n=1,
        regression_kind="r_survey",
    )
    # Compare
    compare_result(loaded_result, python_result, r_result)


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
    df = clarite.load.from_csv(DATA_PATH / "apipop_data.csv", index_col=None)
    # Process data
    df = clarite.modify.make_continuous(df, only=["api00", "ell", "meals", "mobility"])
    df = clarite.modify.colfilter(df, only=["api00", "ell", "meals", "mobility"])
    # Get results
    loaded_result = load_surveylib_results(RESULT_PATH / "api_apipop_result.csv")
    python_result = pd.concat(
        [
            clarite.analyze.ewas(
                outcome="api00", covariates=["meals", "mobility"], data=df, min_n=1
            ),
            clarite.analyze.ewas(
                outcome="api00", covariates=["ell", "mobility"], data=df, min_n=1
            ),
            clarite.analyze.ewas(
                outcome="api00", covariates=["ell", "meals"], data=df, min_n=1
            ),
        ],
        axis=0,
    )
    r_result = pd.concat(
        [
            clarite.analyze.ewas(
                outcome="api00",
                covariates=["meals", "mobility"],
                data=df,
                min_n=1,
                regression_kind="r_survey",
            ),
            clarite.analyze.ewas(
                outcome="api00",
                covariates=["ell", "mobility"],
                data=df,
                min_n=1,
                regression_kind="r_survey",
            ),
            clarite.analyze.ewas(
                outcome="api00",
                covariates=["ell", "meals"],
                data=df,
                min_n=1,
                regression_kind="r_survey",
            ),
        ],
        axis=0,
    )
    # Compare
    compare_result(loaded_result, python_result, r_result)


def test_api_noweights_withNA():
    """Test the api dataset (with na) with no survey info"""
    # Load the data
    df = clarite.load.from_csv(DATA_PATH / "apipop_withna_data.csv", index_col=None)
    # Process data
    df = clarite.modify.make_continuous(df, only=["api00", "ell", "meals", "mobility"])
    df = clarite.modify.colfilter(df, only=["api00", "ell", "meals", "mobility"])
    # Get Results
    loaded_result = load_surveylib_results(RESULT_PATH / "api_apipop_withna_result.csv")
    python_result = pd.concat(
        [
            clarite.analyze.ewas(
                outcome="api00", covariates=["meals", "mobility"], data=df, min_n=1
            ),
            clarite.analyze.ewas(
                outcome="api00", covariates=["ell", "mobility"], data=df, min_n=1
            ),
            clarite.analyze.ewas(
                outcome="api00", covariates=["ell", "meals"], data=df, min_n=1
            ),
        ],
        axis=0,
    )
    r_result = pd.concat(
        [
            clarite.analyze.ewas(
                outcome="api00",
                covariates=["meals", "mobility"],
                data=df,
                min_n=1,
                regression_kind="r_survey",
            ),
            clarite.analyze.ewas(
                outcome="api00",
                covariates=["ell", "mobility"],
                data=df,
                min_n=1,
                regression_kind="r_survey",
            ),
            clarite.analyze.ewas(
                outcome="api00",
                covariates=["ell", "meals"],
                data=df,
                min_n=1,
                regression_kind="r_survey",
            ),
        ],
        axis=0,
    )
    # Compare
    compare_result(loaded_result, python_result, r_result)


def test_api_stratified():
    """Test the api dataset with weights, strata, and fpc"""
    # Load the data
    df = clarite.load.from_csv(DATA_PATH / "apistrat_data.csv", index_col=None)
    # Process data
    df = clarite.modify.make_continuous(df, only=["api00", "ell", "meals", "mobility"])
    design = clarite.survey.SurveyDesignSpec(
        df, weights="pw", cluster=None, strata="stype", fpc="fpc"
    )
    df = clarite.modify.colfilter(df, only=["api00", "ell", "meals", "mobility"])
    # Get Results
    loaded_result = load_surveylib_results(RESULT_PATH / "api_apistrat_result.csv")
    python_result = pd.concat(
        [
            clarite.analyze.ewas(
                outcome="api00",
                covariates=["meals", "mobility"],
                data=df,
                survey_design_spec=design,
                min_n=1,
            ),
            clarite.analyze.ewas(
                outcome="api00",
                covariates=["ell", "mobility"],
                data=df,
                survey_design_spec=design,
                min_n=1,
            ),
            clarite.analyze.ewas(
                outcome="api00",
                covariates=["ell", "meals"],
                data=df,
                survey_design_spec=design,
                min_n=1,
            ),
        ],
        axis=0,
    )
    r_result = pd.concat(
        [
            clarite.analyze.ewas(
                outcome="api00",
                covariates=["meals", "mobility"],
                data=df,
                survey_design_spec=design,
                min_n=1,
                regression_kind="r_survey",
            ),
            clarite.analyze.ewas(
                outcome="api00",
                covariates=["ell", "mobility"],
                data=df,
                survey_design_spec=design,
                min_n=1,
                regression_kind="r_survey",
            ),
            clarite.analyze.ewas(
                outcome="api00",
                covariates=["ell", "meals"],
                data=df,
                survey_design_spec=design,
                min_n=1,
                regression_kind="r_survey",
            ),
        ],
        axis=0,
    )
    # Compare
    compare_result(loaded_result, python_result, r_result)


def test_api_cluster():
    """Test the api dataset with weights, clusters, and fpc"""
    # Load the data
    df = clarite.load.from_csv(DATA_PATH / "apiclus1_data.csv", index_col=None)
    # Process data
    df = clarite.modify.make_continuous(df, only=["api00", "ell", "meals", "mobility"])
    design = clarite.survey.SurveyDesignSpec(
        df, weights="pw", cluster="dnum", strata=None, fpc="fpc"
    )
    df = clarite.modify.colfilter(df, only=["api00", "ell", "meals", "mobility"])
    # Get Results
    loaded_result = load_surveylib_results(RESULT_PATH / "api_apiclus1_result.csv")
    python_result = pd.concat(
        [
            clarite.analyze.ewas(
                outcome="api00",
                covariates=["meals", "mobility"],
                data=df,
                survey_design_spec=design,
                min_n=1,
            ),
            clarite.analyze.ewas(
                outcome="api00",
                covariates=["ell", "mobility"],
                data=df,
                survey_design_spec=design,
                min_n=1,
            ),
            clarite.analyze.ewas(
                outcome="api00",
                covariates=["ell", "meals"],
                data=df,
                survey_design_spec=design,
                min_n=1,
            ),
        ],
        axis=0,
    )
    r_result = pd.concat(
        [
            clarite.analyze.ewas(
                outcome="api00",
                covariates=["meals", "mobility"],
                data=df,
                survey_design_spec=design,
                min_n=1,
                regression_kind="r_survey",
            ),
            clarite.analyze.ewas(
                outcome="api00",
                covariates=["ell", "mobility"],
                data=df,
                survey_design_spec=design,
                min_n=1,
                regression_kind="r_survey",
            ),
            clarite.analyze.ewas(
                outcome="api00",
                covariates=["ell", "meals"],
                data=df,
                survey_design_spec=design,
                min_n=1,
                regression_kind="r_survey",
            ),
        ],
        axis=0,
    )
    # Compare
    compare_result(loaded_result, python_result, r_result)


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


def test_nhanes_noweights(data_NHANES):
    """Test the nhanes dataset with no survey info"""
    # Process data
    df = clarite.modify.colfilter(
        data_NHANES, only=["HI_CHOL", "RIAGENDR", "race", "agecat"]
    )
    # Get Results
    loaded_result = load_surveylib_results(RESULT_PATH / "nhanes_noweights_result.csv")
    python_result = pd.concat(
        [
            clarite.analyze.ewas(
                outcome="HI_CHOL", covariates=["agecat", "RIAGENDR"], data=df
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL", covariates=["race", "RIAGENDR"], data=df
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL", covariates=["race", "agecat"], data=df
            ),
        ],
        axis=0,
    )
    r_result = pd.concat(
        [
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["agecat", "RIAGENDR"],
                data=df,
                regression_kind="r_survey",
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "RIAGENDR"],
                data=df,
                regression_kind="r_survey",
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "agecat"],
                data=df,
                regression_kind="r_survey",
            ),
        ],
        axis=0,
    )
    # Compare
    compare_result(loaded_result, python_result, r_result)


def test_nhanes_noweights_withNA(data_NHANES_withNA):
    """Test the nhanes dataset with no survey info and some missing values in a categorical"""
    # Process data
    df = clarite.modify.colfilter(
        data_NHANES_withNA, only=["HI_CHOL", "RIAGENDR", "race", "agecat"]
    )
    # Get Results
    loaded_result = load_surveylib_results(
        RESULT_PATH / "nhanes_noweights_withna_result.csv"
    )
    python_result = pd.concat(
        [
            clarite.analyze.ewas(
                outcome="HI_CHOL", covariates=["agecat", "RIAGENDR"], data=df
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL", covariates=["race", "RIAGENDR"], data=df
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL", covariates=["race", "agecat"], data=df
            ),
        ],
        axis=0,
    )
    r_result = pd.concat(
        [
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["agecat", "RIAGENDR"],
                data=df,
                regression_kind="r_survey",
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "RIAGENDR"],
                data=df,
                regression_kind="r_survey",
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "agecat"],
                data=df,
                regression_kind="r_survey",
            ),
        ],
        axis=0,
    )
    # Compare
    compare_result(loaded_result, python_result, r_result)


def test_nhanes_fulldesign(data_NHANES):
    """Test the nhanes dataset with the full survey design"""
    # Make Design
    design = clarite.survey.SurveyDesignSpec(
        data_NHANES,
        weights="WTMEC2YR",
        cluster="SDMVPSU",
        strata="SDMVSTRA",
        fpc=None,
        nest=True,
    )
    df = clarite.modify.colfilter(
        data_NHANES, only=["HI_CHOL", "RIAGENDR", "race", "agecat"]
    )
    # Get Results
    loaded_result = load_surveylib_results(RESULT_PATH / "nhanes_complete_result.csv")
    python_result = pd.concat(
        [
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["agecat", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "agecat"],
                data=df,
                survey_design_spec=design,
            ),
        ],
        axis=0,
    )
    r_result = pd.concat(
        [
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["agecat", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
                regression_kind="r_survey",
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
                regression_kind="r_survey",
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "agecat"],
                data=df,
                survey_design_spec=design,
                regression_kind="r_survey",
            ),
        ],
        axis=0,
    )
    # Compare
    compare_result(loaded_result, python_result, r_result)


def test_nhanes_fulldesign_withna(data_NHANES_withNA):
    """Test the nhanes dataset with the full survey design"""
    # Make Design
    design = clarite.survey.SurveyDesignSpec(
        data_NHANES_withNA,
        weights="WTMEC2YR",
        cluster="SDMVPSU",
        strata="SDMVSTRA",
        fpc=None,
        nest=True,
    )
    df = clarite.modify.colfilter(
        data_NHANES_withNA, only=["HI_CHOL", "RIAGENDR", "race", "agecat"]
    )
    # Get Results
    loaded_result = load_surveylib_results(
        RESULT_PATH / "nhanes_complete_withna_result.csv"
    )
    python_result = pd.concat(
        [
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["agecat", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "agecat"],
                data=df,
                survey_design_spec=design,
            ),
        ],
        axis=0,
    )
    r_result = pd.concat(
        [
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["agecat", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
                regression_kind="r_survey",
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
                regression_kind="r_survey",
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "agecat"],
                data=df,
                survey_design_spec=design,
                regression_kind="r_survey",
            ),
        ],
        axis=0,
    )
    # Compare
    compare_result(loaded_result, python_result, r_result)


def test_nhanes_fulldesign_subset_category(data_NHANES):
    """Test the nhanes dataset with the full survey design, subset by dropping a category"""
    # Make Design
    design = clarite.survey.SurveyDesignSpec(
        data_NHANES,
        weights="WTMEC2YR",
        cluster="SDMVPSU",
        strata="SDMVSTRA",
        fpc=None,
        nest=True,
    )
    design.subset(data_NHANES["agecat"] != "(19,39]")
    df = clarite.modify.colfilter(
        data_NHANES, only=["HI_CHOL", "RIAGENDR", "race", "agecat"]
    )
    # Get Results
    loaded_result = load_surveylib_results(
        RESULT_PATH / "nhanes_complete_result_subset_cat.csv"
    )
    python_result = pd.concat(
        [
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["agecat", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "agecat"],
                data=df,
                survey_design_spec=design,
            ),
        ],
        axis=0,
    )
    r_result = pd.concat(
        [
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["agecat", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
                regression_kind="r_survey",
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
                regression_kind="r_survey",
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "agecat"],
                data=df,
                survey_design_spec=design,
                regression_kind="r_survey",
            ),
        ],
        axis=0,
    )
    # Compare
    compare_result(loaded_result, python_result, r_result, rtol=1e-03)


def test_nhanes_fulldesign_subset_continuous():
    """Test the nhanes dataset with the full survey design and a random subset"""
    # Load the data
    df = clarite.load.from_csv(DATA_PATH / "nhanes_data_subset.csv", index_col=None)
    # Process data
    df = clarite.modify.make_binary(df, only=["HI_CHOL", "RIAGENDR"])
    df = clarite.modify.make_categorical(df, only=["race", "agecat"])
    design = clarite.survey.SurveyDesignSpec(
        df,
        weights="WTMEC2YR",
        cluster="SDMVPSU",
        strata="SDMVSTRA",
        fpc=None,
        nest=True,
        drop_unweighted=True,
    )
    design.subset(df["subset"] > 0)
    df = df.drop(columns=["subset"])
    df = clarite.modify.colfilter(df, only=["HI_CHOL", "RIAGENDR", "race", "agecat"])
    # Get Results
    loaded_result = load_surveylib_results(
        RESULT_PATH / "nhanes_complete_result_subset_cont.csv"
    )
    python_result = pd.concat(
        [
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["agecat", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "agecat"],
                data=df,
                survey_design_spec=design,
            ),
        ],
        axis=0,
    )
    r_result = pd.concat(
        [
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["agecat", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
                regression_kind="r_survey",
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
                regression_kind="r_survey",
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "agecat"],
                data=df,
                survey_design_spec=design,
                regression_kind="r_survey",
            ),
        ],
        axis=0,
    )
    # Compare
    compare_result(loaded_result, python_result, r_result)


def test_nhanes_weightsonly(data_NHANES):
    """Test the nhanes dataset with only weights in the survey design"""
    # Make Design
    design = clarite.survey.SurveyDesignSpec(data_NHANES, weights="WTMEC2YR")
    df = clarite.modify.colfilter(
        data_NHANES, only=["HI_CHOL", "RIAGENDR", "race", "agecat"]
    )
    # Get Results
    loaded_result = load_surveylib_results(
        RESULT_PATH / "nhanes_weightsonly_result.csv"
    )
    python_result = pd.concat(
        [
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["agecat", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "agecat"],
                data=df,
                survey_design_spec=design,
            ),
        ],
        axis=0,
    )
    r_result = pd.concat(
        [
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["agecat", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
                regression_kind="r_survey",
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
                regression_kind="r_survey",
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "agecat"],
                data=df,
                survey_design_spec=design,
                regression_kind="r_survey",
            ),
        ],
        axis=0,
    )
    # Compare
    compare_result(loaded_result, python_result, r_result)


def test_nhanes_lonely_certainty(data_NHANES_lonely):
    """Test the nhanes dataset with a lonely PSU and the value set to certainty"""
    # Make Design
    design = clarite.survey.SurveyDesignSpec(
        data_NHANES_lonely,
        weights="WTMEC2YR",
        cluster="SDMVPSU",
        strata="SDMVSTRA",
        fpc=None,
        nest=True,
        single_cluster="certainty",
    )
    df = clarite.modify.colfilter(
        data_NHANES_lonely, only=["HI_CHOL", "RIAGENDR", "race", "agecat"]
    )
    # Get Results
    loaded_result = load_surveylib_results(RESULT_PATH / "nhanes_certainty_result.csv")
    python_result = pd.concat(
        [
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["agecat", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "agecat"],
                data=df,
                survey_design_spec=design,
            ),
        ],
        axis=0,
    )
    r_result = pd.concat(
        [
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["agecat", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
                regression_kind="r_survey",
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
                regression_kind="r_survey",
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "agecat"],
                data=df,
                survey_design_spec=design,
                regression_kind="r_survey",
            ),
        ],
        axis=0,
    )
    # Compare
    compare_result(loaded_result, python_result, r_result)


def test_nhanes_lonely_adjust(data_NHANES_lonely):
    """Test the nhanes dataset with a lonely PSU and the value set to adjust"""
    # Make Design
    design = clarite.survey.SurveyDesignSpec(
        data_NHANES_lonely,
        weights="WTMEC2YR",
        cluster="SDMVPSU",
        strata="SDMVSTRA",
        fpc=None,
        nest=True,
        single_cluster="adjust",
    )
    df = clarite.modify.colfilter(
        data_NHANES_lonely, only=["HI_CHOL", "RIAGENDR", "race", "agecat"]
    )
    # Get Results
    loaded_result = load_surveylib_results(RESULT_PATH / "nhanes_adjust_result.csv")
    python_result = pd.concat(
        [
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["agecat", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "agecat"],
                data=df,
                survey_design_spec=design,
            ),
        ],
        axis=0,
    )
    r_result = pd.concat(
        [
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["agecat", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
                regression_kind="r_survey",
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
                regression_kind="r_survey",
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "agecat"],
                data=df,
                survey_design_spec=design,
                regression_kind="r_survey",
            ),
        ],
        axis=0,
    )
    # Compare
    compare_result(loaded_result, python_result, r_result)


def test_nhanes_lonely_average(data_NHANES_lonely):
    """Test the nhanes dataset with a lonely PSU and the value set to average"""
    # Make Design
    design = clarite.survey.SurveyDesignSpec(
        data_NHANES_lonely,
        weights="WTMEC2YR",
        cluster="SDMVPSU",
        strata="SDMVSTRA",
        fpc=None,
        nest=True,
        single_cluster="average",
    )
    df = clarite.modify.colfilter(
        data_NHANES_lonely, only=["HI_CHOL", "RIAGENDR", "race", "agecat"]
    )
    # Get Results
    loaded_result = load_surveylib_results(RESULT_PATH / "nhanes_average_result.csv")
    python_result = pd.concat(
        [
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["agecat", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "agecat"],
                data=df,
                survey_design_spec=design,
            ),
        ],
        axis=0,
    )
    r_result = pd.concat(
        [
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["agecat", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
                regression_kind="r_survey",
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
                regression_kind="r_survey",
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "agecat"],
                data=df,
                survey_design_spec=design,
                regression_kind="r_survey",
            ),
        ],
        axis=0,
    )
    # Compare
    compare_result(loaded_result, python_result, r_result)


def test_nhanes_realistic():
    """Test a more realistic set of NHANES data, specifically using multiple weights and missing values"""
    # Load the data
    df = clarite.load.from_tsv(
        DATA_PATH.parent / "test_data_files" / "nhanes_real.txt", index_col="ID"
    )
    # Process data
    # Split out survey info
    survey_cols = ["SDMVPSU", "SDMVSTRA", "WTMEC4YR", "WTSHM4YR", "WTSVOC4Y"]
    survey_df = df[survey_cols]
    df = df.drop(columns=survey_cols)

    df = clarite.modify.make_binary(
        df,
        only=[
            "RHQ570",
            "first_degree_support",
            "SDDSRVYR",
            "female",
            "black",
            "mexican",
            "other_hispanic",
            "other_eth",
        ],
    )
    df = clarite.modify.make_categorical(df, only=["SES_LEVEL"])
    design = clarite.survey.SurveyDesignSpec(
        survey_df,
        weights={
            "RHQ570": "WTMEC4YR",
            "first_degree_support": "WTMEC4YR",
            "URXUPT": "WTSHM4YR",
            "LBXV3A": "WTSVOC4Y",
            "LBXBEC": "WTMEC4YR",
        },
        cluster="SDMVPSU",
        strata="SDMVSTRA",
        fpc=None,
        nest=True,
    )
    # Get Results
    loaded_result = load_surveylib_results(RESULT_PATH / "nhanes_real_result.csv")
    python_result = clarite.analyze.ewas(
        outcome="BMXBMI",
        covariates=[
            "SES_LEVEL",
            "SDDSRVYR",
            "female",
            "black",
            "mexican",
            "other_hispanic",
            "other_eth",
            "RIDAGEYR",
        ],
        data=df,
        survey_design_spec=design,
    )
    r_result = clarite.analyze.ewas(
        outcome="BMXBMI",
        covariates=[
            "SES_LEVEL",
            "SDDSRVYR",
            "female",
            "black",
            "mexican",
            "other_hispanic",
            "other_eth",
            "RIDAGEYR",
        ],
        data=df,
        survey_design_spec=design,
        regression_kind="r_survey",
    )
    # Compare
    compare_result(loaded_result, python_result, r_result)


def test_nhanes_subset_singleclusters():
    """Test a partial nhanes dataset with the full survey design with a subset causing single clusters"""
    # Load the data
    df = clarite.load.from_tsv(
        DATA_PATH.parent / "test_data_files" / "nhanes_subset" / "data.txt"
    )
    survey_df = clarite.load.from_tsv(
        DATA_PATH.parent / "test_data_files" / "nhanes_subset" / "design_data.txt"
    )
    survey_df = survey_df.loc[df.index]
    # Process data
    df = clarite.modify.make_binary(df, only=["LBXHBC", "black", "female"])
    df = clarite.modify.make_categorical(df, only=["SES_LEVEL", "SDDSRVYR"])
    # Create design
    design = clarite.survey.SurveyDesignSpec(
        survey_df,
        weights="WTMEC4YR",
        cluster="SDMVPSU",
        strata="SDMVSTRA",
        fpc=None,
        nest=True,
    )
    design.subset(df["black"] == 1)
    df = df.drop(columns="black")
    # Get Results
    loaded_result = load_surveylib_results(RESULT_PATH / "nhanes_subset_result.csv")
    covariates = ["female", "SES_LEVEL", "RIDAGEYR", "SDDSRVYR", "BMXBMI"]
    python_result = clarite.analyze.ewas(
        outcome="LBXLYPCT",
        covariates=covariates,
        data=df,
        survey_design_spec=design,
        min_n=50,
    )
    r_result = clarite.analyze.ewas(
        outcome="LBXLYPCT",
        covariates=covariates,
        data=df,
        survey_design_spec=design,
        min_n=50,
        regression_kind="r_survey",
    )
    # Compare
    compare_result(loaded_result, python_result, r_result)
