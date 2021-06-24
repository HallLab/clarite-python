"""
Note: relative tolerance must be set for some tests due to floating value rounding differences between R and Python
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal

import clarite

TESTS_PATH = Path(__file__).parent.parent
DATA_PATH = TESTS_PATH / "test_data_files"
RESULT_PATH = TESTS_PATH / "r_test_output" / "analyze"


def python_cat_to_r_cat(python_df):
    """
    Return the same dataframe with the 3rd level of the multiindex ("Category") updated
    so that the python-style name (e.g. race[T.3]) is converted to r-style (e.g. race3)
    """
    re_str = re.compile(r"(?P<var_name>.+)\[T\.(?P<cat_name>.+)\]")
    df = python_df.reset_index(level="Category", drop=False)
    df["Category"] = df["Category"].apply(
        lambda s: s if re_str.match(s) is None else "".join(re_str.match(s).groups())
    )
    df = df.set_index("Category", append=True)
    return df


def compare_loaded(
    python_result, surveylib_result_file, compare_diffAIC=False, rtol=None
):
    """
    Compare surveylib results (run outside of CLARITE) to CLARITE results.
    Minor fixes for compatibility, such as ignoring order of rows/columns ('check_like')
    """
    # Load results run outside of CLARITE
    loaded_result = pd.read_csv(surveylib_result_file)
    loaded_result = loaded_result.set_index("Variable")
    loaded_result["N"] = loaded_result["N"].astype("Int64")

    # Drop DiffAIC unless a comparison makes sense
    if not compare_diffAIC:
        loaded_result = loaded_result[
            [c for c in loaded_result.columns if c != "Diff_AIC"]
        ]
        python_result = python_result[
            [c for c in python_result.columns if c != "Diff_AIC"]
        ]

    # Format python result to match
    python_result = python_result[loaded_result.columns]
    python_result = python_result.reset_index(level="Outcome", drop=True)

    # Compare
    if rtol is not None:
        assert_frame_equal(python_result, loaded_result, rtol=rtol, check_like=True)
    else:
        assert_frame_equal(python_result, loaded_result, check_like=True)


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


@pytest.mark.parametrize(
    "design_str,standardize",
    [
        ("withoutfpc", False),
        ("withoutfpc", True),
        ("withfpc", False),
        ("withfpc", True),
        ("nostrata", False),
        ("nostrata", True),
    ],
)
def test_fpc(data_fpc, design_str, standardize):
    """Use a survey design specifying weights, cluster, strata"""
    # Set data and design for each test
    if design_str == "withoutfpc":
        design = clarite.survey.SurveyDesignSpec(
            data_fpc, weights="weight", cluster="psuid", strata="stratid", nest=True
        )
        df = clarite.modify.colfilter(data_fpc, only=["x", "y"])
        surveylib_result_file = RESULT_PATH / "fpc_withoutfpc_result.csv"
    elif design_str == "withfpc":
        design = clarite.survey.SurveyDesignSpec(
            data_fpc,
            weights="weight",
            cluster="psuid",
            strata="stratid",
            fpc="Nh",
            nest=True,
        )
        df = clarite.modify.colfilter(data_fpc, only=["x", "y"])
        surveylib_result_file = RESULT_PATH / "fpc_withfpc_result.csv"
    elif design_str == "nostrata":
        # Load the data
        df = clarite.load.from_csv(DATA_PATH / "fpc_nostrat_data.csv", index_col=None)
        # Process data
        df = clarite.modify.make_continuous(df, only=["x", "y"])
        design = clarite.survey.SurveyDesignSpec(
            df, weights="weight", cluster="psuid", strata=None, fpc="Nh", nest=True
        )
        df = clarite.modify.colfilter(df, only=["x", "y"])
        surveylib_result_file = RESULT_PATH / "fpc_withfpc_nostrat_result.csv"
    else:
        raise ValueError(f"design_str unknown: '{design_str}'")

    # Get results
    python_result = clarite.analyze.ewas(
        outcome="y",
        covariates=[],
        data=df,
        survey_design_spec=design,
        min_n=1,
        standardize_data=standardize,
    )
    r_result = clarite.analyze.ewas(
        outcome="y",
        covariates=[],
        data=df,
        survey_design_spec=design,
        min_n=1,
        regression_kind="r_survey",
        standardize_data=standardize,
    )
    # Compare
    if not standardize:
        compare_loaded(python_result, surveylib_result_file)
    assert_frame_equal(python_result, r_result)


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


@pytest.mark.parametrize(
    "design_str,standardize",
    [
        ("noweights", False),
        ("noweights", True),
        ("noweights_withNA", False),
        ("noweights_withNA", True),
        ("stratified", False),
        ("stratified", True),
        ("cluster", False),
        ("cluster", True),
    ],
)
def test_api(design_str, standardize):
    """Test the api dataset with no survey info"""
    if design_str == "noweights":
        df = clarite.load.from_csv(DATA_PATH / "apipop_data.csv", index_col=None)
        df = clarite.modify.make_continuous(
            df, only=["api00", "ell", "meals", "mobility"]
        )
        df = clarite.modify.colfilter(df, only=["api00", "ell", "meals", "mobility"])
        design = None
        surveylib_result_file = RESULT_PATH / "api_apipop_result.csv"
    elif design_str == "noweights_withNA":
        df = clarite.load.from_csv(DATA_PATH / "apipop_withna_data.csv", index_col=None)
        df = clarite.modify.make_continuous(
            df, only=["api00", "ell", "meals", "mobility"]
        )
        df = clarite.modify.colfilter(df, only=["api00", "ell", "meals", "mobility"])
        design = None
        surveylib_result_file = RESULT_PATH / "api_apipop_withna_result.csv"
    elif design_str == "stratified":
        df = clarite.load.from_csv(DATA_PATH / "apistrat_data.csv", index_col=None)
        df = clarite.modify.make_continuous(
            df, only=["api00", "ell", "meals", "mobility"]
        )
        design = clarite.survey.SurveyDesignSpec(
            df, weights="pw", cluster=None, strata="stype", fpc="fpc"
        )
        df = clarite.modify.colfilter(df, only=["api00", "ell", "meals", "mobility"])
        surveylib_result_file = RESULT_PATH / "api_apistrat_result.csv"
    elif design_str == "cluster":
        df = clarite.load.from_csv(DATA_PATH / "apiclus1_data.csv", index_col=None)
        df = clarite.modify.make_continuous(
            df, only=["api00", "ell", "meals", "mobility"]
        )
        design = clarite.survey.SurveyDesignSpec(
            df, weights="pw", cluster="dnum", strata=None, fpc="fpc"
        )
        df = clarite.modify.colfilter(df, only=["api00", "ell", "meals", "mobility"])
        surveylib_result_file = RESULT_PATH / "api_apiclus1_result.csv"
    else:
        raise ValueError(f"design_str unknown: '{design_str}'")

    # Run analysis and comparison
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
    if not standardize:
        compare_loaded(python_result, surveylib_result_file)
    assert_frame_equal(python_result, r_result)


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


@pytest.mark.parametrize("standardize", [False, True])
def test_nhanes_noweights(data_NHANES, standardize):
    """Test the nhanes dataset with no survey info"""
    # Process data
    df = clarite.modify.colfilter(
        data_NHANES, only=["HI_CHOL", "RIAGENDR", "race", "agecat"]
    )
    # Get Results
    surveylib_result_file = RESULT_PATH / "nhanes_noweights_result.csv"
    python_result = pd.concat(
        [
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["agecat", "RIAGENDR"],
                data=df,
                standardize_data=standardize,
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "RIAGENDR"],
                data=df,
                standardize_data=standardize,
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "agecat"],
                data=df,
                standardize_data=standardize,
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
                standardize_data=standardize,
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "RIAGENDR"],
                data=df,
                regression_kind="r_survey",
                standardize_data=standardize,
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "agecat"],
                data=df,
                regression_kind="r_survey",
                standardize_data=standardize,
            ),
        ],
        axis=0,
    )
    # Compare
    if not standardize:
        compare_loaded(python_result, surveylib_result_file)
    assert_frame_equal(python_result, r_result)


@pytest.mark.parametrize("standardize", [False, True])
def test_nhanes_noweights_withNA(data_NHANES_withNA, standardize):
    """Test the nhanes dataset with no survey info and some missing values in a categorical"""
    # Process data
    df = clarite.modify.colfilter(
        data_NHANES_withNA, only=["HI_CHOL", "RIAGENDR", "race", "agecat"]
    )
    # Get Results
    surveylib_result_file = RESULT_PATH / "nhanes_noweights_withna_result.csv"
    python_result = pd.concat(
        [
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["agecat", "RIAGENDR"],
                data=df,
                standardize_data=standardize,
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "RIAGENDR"],
                data=df,
                standardize_data=standardize,
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "agecat"],
                data=df,
                standardize_data=standardize,
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
                standardize_data=standardize,
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "RIAGENDR"],
                data=df,
                regression_kind="r_survey",
                standardize_data=standardize,
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "agecat"],
                data=df,
                regression_kind="r_survey",
                standardize_data=standardize,
            ),
        ],
        axis=0,
    )
    # Compare
    if not standardize:
        compare_loaded(python_result, surveylib_result_file)
    assert_frame_equal(python_result, r_result)


@pytest.mark.parametrize("standardize", [False, True])
def test_nhanes_fulldesign(data_NHANES, standardize):
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
    surveylib_result_file = RESULT_PATH / "nhanes_complete_result.csv"
    python_result = pd.concat(
        [
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["agecat", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
                standardize_data=standardize,
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
                standardize_data=standardize,
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "agecat"],
                data=df,
                survey_design_spec=design,
                standardize_data=standardize,
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
                standardize_data=standardize,
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
                regression_kind="r_survey",
                standardize_data=standardize,
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "agecat"],
                data=df,
                survey_design_spec=design,
                regression_kind="r_survey",
                standardize_data=standardize,
            ),
        ],
        axis=0,
    )
    # Compare
    if not standardize:
        # Skip diffAIC due to survey weights
        compare_loaded(
            python_result, surveylib_result_file, compare_diffAIC=False, rtol=1e-04
        )
    assert_frame_equal(python_result, r_result, rtol=1e-04)


@pytest.mark.parametrize("standardize", [False, True])
def test_nhanes_fulldesign_withna(data_NHANES_withNA, standardize):
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
    surveylib_result_file = RESULT_PATH / "nhanes_complete_withna_result.csv"
    python_result = pd.concat(
        [
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["agecat", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
                standardize_data=standardize,
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
                standardize_data=standardize,
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "agecat"],
                data=df,
                survey_design_spec=design,
                standardize_data=standardize,
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
                standardize_data=standardize,
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
                regression_kind="r_survey",
                standardize_data=standardize,
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "agecat"],
                data=df,
                survey_design_spec=design,
                regression_kind="r_survey",
                standardize_data=standardize,
            ),
        ],
        axis=0,
    )
    # Compare
    if not standardize:
        # Skip diffAIC due to survey weights
        compare_loaded(python_result, surveylib_result_file, compare_diffAIC=False)
    assert_frame_equal(python_result, r_result)


@pytest.mark.parametrize("standardize", [False, True])
def test_nhanes_fulldesign_subset_category(data_NHANES, standardize):
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
    surveylib_result_file = RESULT_PATH / "nhanes_complete_result_subset_cat.csv"
    python_result = pd.concat(
        [
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["agecat", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
                standardize_data=standardize,
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
                standardize_data=standardize,
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "agecat"],
                data=df,
                survey_design_spec=design,
                standardize_data=standardize,
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
                standardize_data=standardize,
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
                regression_kind="r_survey",
                standardize_data=standardize,
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "agecat"],
                data=df,
                survey_design_spec=design,
                regression_kind="r_survey",
                standardize_data=standardize,
            ),
        ],
        axis=0,
    )
    # Compare
    if not standardize:
        # Skip diffAIC due to survey weights
        compare_loaded(
            python_result, surveylib_result_file, compare_diffAIC=False, rtol=1e-04
        )
    assert_frame_equal(python_result, r_result, rtol=1e-04)


@pytest.mark.parametrize("standardize", [False, True])
def test_nhanes_fulldesign_subset_continuous(standardize):
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
    surveylib_result_file = RESULT_PATH / "nhanes_complete_result_subset_cont.csv"
    python_result = pd.concat(
        [
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["agecat", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
                standardize_data=standardize,
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
                standardize_data=standardize,
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "agecat"],
                data=df,
                survey_design_spec=design,
                standardize_data=standardize,
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
                standardize_data=standardize,
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
                regression_kind="r_survey",
                standardize_data=standardize,
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "agecat"],
                data=df,
                survey_design_spec=design,
                regression_kind="r_survey",
                standardize_data=standardize,
            ),
        ],
        axis=0,
    )
    # Compare
    if not standardize:
        # Skip diffAIC due to survey weights
        compare_loaded(python_result, surveylib_result_file, compare_diffAIC=False)
    assert_frame_equal(python_result, r_result)


@pytest.mark.parametrize("standardize", [False, True])
def test_nhanes_weightsonly(data_NHANES, standardize):
    """Test the nhanes dataset with only weights in the survey design"""
    # Make Design
    design = clarite.survey.SurveyDesignSpec(data_NHANES, weights="WTMEC2YR")
    df = clarite.modify.colfilter(
        data_NHANES, only=["HI_CHOL", "RIAGENDR", "race", "agecat"]
    )
    # Get Results
    surveylib_result_file = RESULT_PATH / "nhanes_weightsonly_result.csv"
    python_result = pd.concat(
        [
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["agecat", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
                standardize_data=standardize,
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
                standardize_data=standardize,
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "agecat"],
                data=df,
                survey_design_spec=design,
                standardize_data=standardize,
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
                standardize_data=standardize,
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
                regression_kind="r_survey",
                standardize_data=standardize,
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "agecat"],
                data=df,
                survey_design_spec=design,
                regression_kind="r_survey",
                standardize_data=standardize,
            ),
        ],
        axis=0,
    )
    # Compare
    if not standardize:
        # Skip diffAIC due to survey weights
        compare_loaded(python_result, surveylib_result_file, compare_diffAIC=False)
    assert_frame_equal(python_result, r_result)


@pytest.mark.parametrize(
    "single_cluster,load_filename,standardize",
    [
        ("certainty", "nhanes_certainty_result.csv", False),
        ("certainty", "nhanes_certainty_result.csv", True),
        ("adjust", "nhanes_adjust_result.csv", False),
        ("adjust", "nhanes_adjust_result.csv", True),
        ("average", "nhanes_average_result.csv", False),
        ("average", "nhanes_average_result.csv", True),
    ],
)
def test_nhanes_lonely(data_NHANES_lonely, single_cluster, load_filename, standardize):
    """Test the nhanes dataset with a lonely PSU and the value set to certainty"""
    # Make Design
    design = clarite.survey.SurveyDesignSpec(
        data_NHANES_lonely,
        weights="WTMEC2YR",
        cluster="SDMVPSU",
        strata="SDMVSTRA",
        fpc=None,
        nest=True,
        single_cluster=single_cluster,
    )
    df = clarite.modify.colfilter(
        data_NHANES_lonely, only=["HI_CHOL", "RIAGENDR", "race", "agecat"]
    )
    # Get Results
    surveylib_result_file = RESULT_PATH / load_filename
    python_result = pd.concat(
        [
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["agecat", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
                standardize_data=standardize,
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
                standardize_data=standardize,
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "agecat"],
                data=df,
                survey_design_spec=design,
                standardize_data=standardize,
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
                standardize_data=standardize,
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "RIAGENDR"],
                data=df,
                survey_design_spec=design,
                regression_kind="r_survey",
                standardize_data=standardize,
            ),
            clarite.analyze.ewas(
                outcome="HI_CHOL",
                covariates=["race", "agecat"],
                data=df,
                survey_design_spec=design,
                regression_kind="r_survey",
                standardize_data=standardize,
            ),
        ],
        axis=0,
    )
    # Compare
    if not standardize:
        # Skip diffAIC due to survey weights
        compare_loaded(
            python_result, surveylib_result_file, compare_diffAIC=False, rtol=1e-04
        )
    assert_frame_equal(python_result, r_result, rtol=1e-04)


@pytest.mark.parametrize("standardize", [False, True])
def test_nhanes_realistic(standardize):
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
    surveylib_result_file = RESULT_PATH / "nhanes_real_result.csv"
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
        standardize_data=standardize,
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
        standardize_data=standardize,
    )
    # Compare
    if not standardize:
        # Skip diffAIC due to survey weights
        compare_loaded(python_result, surveylib_result_file, compare_diffAIC=False)
        assert_frame_equal(python_result, r_result, rtol=1e-04)
    else:
        assert_frame_equal(python_result, r_result, rtol=1e-02)


@pytest.mark.parametrize("standardize", [False, True])
def test_nhanes_subset_singleclusters(standardize):
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
    surveylib_result_file = RESULT_PATH / "nhanes_subset_result.csv"
    covariates = ["female", "SES_LEVEL", "RIDAGEYR", "SDDSRVYR", "BMXBMI"]
    python_result = clarite.analyze.ewas(
        outcome="LBXLYPCT",
        covariates=covariates,
        data=df,
        survey_design_spec=design,
        min_n=50,
        standardize_data=standardize,
    )
    r_result = clarite.analyze.ewas(
        outcome="LBXLYPCT",
        covariates=covariates,
        data=df,
        survey_design_spec=design,
        min_n=50,
        regression_kind="r_survey",
        standardize_data=standardize,
    )
    # Compare
    if not standardize:
        # Skip diffAIC due to survey weights
        compare_loaded(python_result, surveylib_result_file, compare_diffAIC=False)
        assert_frame_equal(python_result, r_result, rtol=1e-04)
    else:
        assert_frame_equal(python_result, r_result, rtol=1e-02)


def test_report_betas(data_NHANES):
    # Process data
    df = clarite.modify.colfilter(
        data_NHANES, only=["HI_CHOL", "RIAGENDR", "race", "agecat"]
    )
    # Get Results
    normal_result = clarite.analyze.ewas(
        outcome="HI_CHOL", covariates=["agecat", "RIAGENDR"], data=df
    )
    betas_result_python = clarite.analyze.ewas(
        outcome="HI_CHOL",
        covariates=["agecat", "RIAGENDR"],
        data=df,
        report_categorical_betas=True,
    )
    betas_result_r = clarite.analyze.ewas(
        outcome="HI_CHOL",
        covariates=["agecat", "RIAGENDR"],
        data=df,
        regression_kind="r_survey",
        report_categorical_betas=True,
    )
    # Ensure including betas worked
    assert len(betas_result_python) == len(df["race"].cat.categories) - 1
    assert len(betas_result_python) == len(betas_result_r)
    # Ensure including betas did not change other values
    beta_sub = betas_result_python.groupby(level=[0, 1]).first()
    beta_sub[["Beta", "SE", "Beta_pvalue"]] = np.nan
    assert_frame_equal(beta_sub, normal_result)
    # Ensure python and R results match
    betas_result_python = python_cat_to_r_cat(betas_result_python)
    assert_frame_equal(
        betas_result_python,
        betas_result_r,
        check_dtype=False,
        check_exact=False,
        atol=0,
        rtol=1e-04,
    )


def test_report_betas_fulldesign(data_NHANES):
    design = clarite.survey.SurveyDesignSpec(
        data_NHANES,
        weights="WTMEC2YR",
        cluster="SDMVPSU",
        strata="SDMVSTRA",
        fpc=None,
        nest=True,
    )
    # Process data
    df = clarite.modify.colfilter(
        data_NHANES, only=["HI_CHOL", "RIAGENDR", "race", "agecat"]
    )
    # Get Results
    normal_result = clarite.analyze.ewas(
        outcome="HI_CHOL",
        covariates=["agecat", "RIAGENDR"],
        data=df,
        survey_design_spec=design,
    )
    betas_result_python = clarite.analyze.ewas(
        outcome="HI_CHOL",
        covariates=["agecat", "RIAGENDR"],
        data=df,
        report_categorical_betas=True,
        survey_design_spec=design,
    )
    betas_result_r = clarite.analyze.ewas(
        outcome="HI_CHOL",
        covariates=["agecat", "RIAGENDR"],
        data=df,
        report_categorical_betas=True,
        survey_design_spec=design,
        regression_kind="r_survey",
    )
    # Ensure including betas worked
    assert len(betas_result_python) == len(df["race"].cat.categories) - 1
    assert len(betas_result_python) == len(betas_result_r)
    # Ensure including betas did not change other values
    beta_sub = betas_result_python.groupby(level=[0, 1]).first()
    beta_sub[["Beta", "SE", "Beta_pvalue"]] = np.nan
    assert_frame_equal(beta_sub, normal_result)
    # Ensure python and R results match
    betas_result_python = python_cat_to_r_cat(betas_result_python)
    assert_frame_equal(
        betas_result_python,
        betas_result_r,
        check_dtype=False,
        check_exact=False,
        atol=0,
    )
