from pathlib import Path

import numpy as np
import pandas as pd
import clarite

DATA_PATH = Path(__file__).parent.parent / 'r_test_output'


def load_r_results(filename):
    """Load R results and convert column names to match python results"""
    r_result = pd.read_csv(filename)
    r_result = r_result.set_index('Variable')
    r_result[["Beta","SE","Diff_AIC","pvalue"]] = r_result[["Beta","SE","Diff_AIC","pvalue"]].astype('float64')
    return r_result


def compare_result(loaded_r_result, calculated_result, atol=0, rtol=1e-04):
    """Binary variables must be specified, since there are expected differences"""
    # Remove "Phenotype" from the index in calculated results
    calculated_result.reset_index(drop=False).set_index('Variable').drop(columns=['Phenotype'])
    # Merge
    merged = pd.merge(left=loaded_r_result, right=calculated_result,
                      left_index=True, right_index=True,
                      how="inner", suffixes=("_loaded", "_calculated"))
    try:
        assert len(merged) == len(loaded_r_result) == len(calculated_result)
    except AssertionError:
        raise ValueError(f" Loaded Results have {len(loaded_r_result):,} rows,"
                         f" Calculated results have {len(calculated_result):,} rows,"
                         f" merged data has {len(merged):,} rows")
    # Close-enough equality of numeric values
    for var in ["N", "Beta", "SE", "pvalue"]:
        try:
            assert np.allclose(merged[f"{var}_loaded"], merged[f"{var}_calculated"],
                               equal_nan=True, atol=atol, rtol=rtol)
        except AssertionError:
            raise ValueError(f"{var}:\n"
                             f"{merged[f'{var}_loaded']}\n"
                             f"{merged[f'{var}_calculated']}")
    for var in ["Diff_AIC"]:
        # Pass if loaded result is NaN (quasibinomial) or calculated result is NaN (survey data used)
        either_nan = merged[[f'{var}_loaded', f'{var}_calculated']].isna().any(axis=1)
        try:
            # Value must be close when both exist or both are NaN
            assert np.allclose(merged.loc[~either_nan, f"{var}_loaded"],
                               merged.loc[~either_nan, f"{var}_calculated"], equal_nan=True)
        except AssertionError:
            raise ValueError(f"{var}: Loaded ({merged[f'{var}_loaded']}) != Calculated ({merged[f'{var}_calculated']})")

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


def test_fpc_withoutfpc():
    """Use a survey design specifying weights, cluster, strata"""
    # Load the data
    df = clarite.load.from_csv(DATA_PATH / "fpc_data.csv", index_col=None)
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
    df = clarite.load.from_csv(DATA_PATH / "fpc_data.csv", index_col=None)
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
    df = clarite.load.from_csv(DATA_PATH / "fpc_nostrat_data.csv", index_col=None)
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
    df = clarite.load.from_csv(DATA_PATH / "apipop_data.csv", index_col=None)
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


def test_api_noweights_withNA():
    """Test the api dataset (with na) with no survey info"""
    # Load the data
    df = clarite.load.from_csv(DATA_PATH / "apipop_withna_data.csv", index_col=None)
    # Load the expected results
    r_result = load_r_results(DATA_PATH / "api_apipop_withna_result.csv")
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
    df = clarite.load.from_csv(DATA_PATH / "apistrat_data.csv", index_col=None)
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
    df = clarite.load.from_csv(DATA_PATH / "apiclus1_data.csv", index_col=None)
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

# Note that some tests are given wide tolerances to pass:
#  -


def test_nhanes_noweights():
    """Test the nhanes dataset with no survey info"""
    # Load the data
    df = clarite.load.from_csv(DATA_PATH / "nhanes_data.csv", index_col=None)
    # Load the expected results
    r_result = load_r_results(DATA_PATH / "nhanes_noweights_result.csv")
    # Process data
    df = clarite.modify.make_binary(df, only=["HI_CHOL", "RIAGENDR"])
    df = clarite.modify.make_categorical(df, only=["race", "agecat"])
    df = clarite.modify.colfilter(df, only=["HI_CHOL", "RIAGENDR", "race", "agecat"])
    python_result = pd.concat([
        clarite.analyze.ewas(phenotype="HI_CHOL", covariates=["agecat", "RIAGENDR"], data=df),
        clarite.analyze.ewas(phenotype="HI_CHOL", covariates=["race", "RIAGENDR"], data=df),
        clarite.analyze.ewas(phenotype="HI_CHOL", covariates=["race", "agecat"], data=df),
        ], axis=0)
    # Compare
    compare_result(r_result, python_result)


def test_nhanes_noweights_withNA():
    """Test the nhanes dataset with no survey info and some missing values in a categorical"""
    # Load the data
    df = clarite.load.from_csv(DATA_PATH / "nhanes_NAs_data.csv", index_col=None)
    # Load the expected results
    r_result = load_r_results(DATA_PATH / "nhanes_noweights_withna_result.csv")
    # Process data
    df = clarite.modify.make_binary(df, only=["HI_CHOL", "RIAGENDR"])
    df = clarite.modify.make_categorical(df, only=["race", "agecat"])
    df = clarite.modify.colfilter(df, only=["HI_CHOL", "RIAGENDR", "race", "agecat"])
    python_result = pd.concat([
        clarite.analyze.ewas(phenotype="HI_CHOL", covariates=["agecat", "RIAGENDR"], data=df),
        clarite.analyze.ewas(phenotype="HI_CHOL", covariates=["race", "RIAGENDR"], data=df),
        clarite.analyze.ewas(phenotype="HI_CHOL", covariates=["race", "agecat"], data=df),
        ], axis=0)
    # Compare
    compare_result(r_result, python_result)


def test_nhanes_fulldesign():
    """Test the nhanes dataset with the full survey design"""
    # Load the data
    df = clarite.load.from_csv(DATA_PATH / "nhanes_data.csv", index_col=None)
    # Load the expected results
    r_result = load_r_results(DATA_PATH / "nhanes_complete_result.csv")
    # Process data
    df = clarite.modify.make_binary(df, only=["HI_CHOL", "RIAGENDR"])
    df = clarite.modify.make_categorical(df, only=["race", "agecat"])
    design = clarite.survey.SurveyDesignSpec(df, weights="WTMEC2YR", cluster="SDMVPSU", strata="SDMVSTRA",
                                             fpc=None, nest=True)
    df = clarite.modify.colfilter(df, only=["HI_CHOL", "RIAGENDR", "race", "agecat"])
    python_result = pd.concat([
        clarite.analyze.ewas(phenotype="HI_CHOL", covariates=["agecat", "RIAGENDR"], data=df,
                             survey_design_spec=design),
        clarite.analyze.ewas(phenotype="HI_CHOL", covariates=["race", "RIAGENDR"], data=df,
                             survey_design_spec=design),
        clarite.analyze.ewas(phenotype="HI_CHOL", covariates=["race", "agecat"], data=df,
                             survey_design_spec=design),
        ], axis=0)
    # Compare
    compare_result(r_result, python_result)


def test_nhanes_fulldesign_withna():
    """Test the nhanes dataset with the full survey design"""
    # Load the data
    df = clarite.load.from_csv(DATA_PATH / "nhanes_NAs_data.csv", index_col=None)
    # Load the expected results
    r_result = load_r_results(DATA_PATH / "nhanes_complete_withna_result.csv")
    # Process data
    df = clarite.modify.make_binary(df, only=["HI_CHOL", "RIAGENDR"])
    df = clarite.modify.make_categorical(df, only=["race", "agecat"])
    design = clarite.survey.SurveyDesignSpec(df, weights="WTMEC2YR", cluster="SDMVPSU", strata="SDMVSTRA",
                                             fpc=None, nest=True)
    df = clarite.modify.colfilter(df, only=["HI_CHOL", "RIAGENDR", "race", "agecat"])
    python_result = pd.concat([
        clarite.analyze.ewas(phenotype="HI_CHOL", covariates=["agecat", "RIAGENDR"], data=df,
                             survey_design_spec=design),
        clarite.analyze.ewas(phenotype="HI_CHOL", covariates=["race", "RIAGENDR"], data=df,
                             survey_design_spec=design),
        clarite.analyze.ewas(phenotype="HI_CHOL", covariates=["race", "agecat"], data=df,
                             survey_design_spec=design),
        ], axis=0)
    # Compare
    compare_result(r_result, python_result)


def test_nhanes_fulldesign_subset_category():
    """Test the nhanes dataset with the full survey design, subset by dropping a category"""
    # Load the data
    df = clarite.load.from_csv(DATA_PATH / "nhanes_data.csv", index_col=None)
    # Load the expected results
    r_result = load_r_results(DATA_PATH / "nhanes_complete_result_subset_cat.csv")
    # Process data
    df = clarite.modify.make_binary(df, only=["HI_CHOL", "RIAGENDR"])
    df = clarite.modify.make_categorical(df, only=["race", "agecat"])
    design = clarite.survey.SurveyDesignSpec(df, weights="WTMEC2YR", cluster="SDMVPSU", strata="SDMVSTRA",
                                             fpc=None, nest=True, drop_unweighted=True)
    design.subset(df['agecat'] != "(19,39]")
    df = clarite.modify.colfilter(df, only=["HI_CHOL", "RIAGENDR", "race", "agecat"])
    python_result = pd.concat([
        clarite.analyze.ewas(phenotype="HI_CHOL", covariates=["agecat", "RIAGENDR"], data=df,
                             survey_design_spec=design),
        clarite.analyze.ewas(phenotype="HI_CHOL", covariates=["race", "RIAGENDR"], data=df,
                             survey_design_spec=design),
        clarite.analyze.ewas(phenotype="HI_CHOL", covariates=["race", "agecat"], data=df,
                             survey_design_spec=design),
        ], axis=0)
    # Compare
    compare_result(r_result, python_result, rtol=1e-03)


def test_nhanes_fulldesign_subset_continuous():
    """Test the nhanes dataset with the full survey design and a random subset"""
    # Load the data
    df = clarite.load.from_csv(DATA_PATH / "nhanes_data_subset.csv", index_col=None)
    # Load the expected results
    r_result = load_r_results(DATA_PATH / "nhanes_complete_result_subset_cont.csv")
    # Process data
    df = clarite.modify.make_binary(df, only=["HI_CHOL", "RIAGENDR"])
    df = clarite.modify.make_categorical(df, only=["race", "agecat"])
    design = clarite.survey.SurveyDesignSpec(df, weights="WTMEC2YR", cluster="SDMVPSU", strata="SDMVSTRA",
                                             fpc=None, nest=True, drop_unweighted=True)
    design.subset(df['subset'] > 0)
    df = df.drop(columns=['subset'])
    df = clarite.modify.colfilter(df, only=["HI_CHOL", "RIAGENDR", "race", "agecat"])
    python_result = pd.concat([
        clarite.analyze.ewas(phenotype="HI_CHOL", covariates=["agecat", "RIAGENDR"], data=df,
                             survey_design_spec=design),
        clarite.analyze.ewas(phenotype="HI_CHOL", covariates=["race", "RIAGENDR"], data=df,
                             survey_design_spec=design),
        clarite.analyze.ewas(phenotype="HI_CHOL", covariates=["race", "agecat"], data=df,
                             survey_design_spec=design),
        ], axis=0)
    # Compare
    compare_result(r_result, python_result)


def test_nhanes_weightsonly():
    """Test the nhanes dataset with only weights in the survey design"""
    # Load the data
    df = clarite.load.from_csv(DATA_PATH / "nhanes_data.csv", index_col=None)
    # Load the expected results
    r_result = load_r_results(DATA_PATH / "nhanes_weightsonly_result.csv")
    # Process data
    df = clarite.modify.make_binary(df, only=["HI_CHOL", "RIAGENDR"])
    df = clarite.modify.make_categorical(df, only=["race", "agecat"])
    design = clarite.survey.SurveyDesignSpec(df, weights="WTMEC2YR")
    df = clarite.modify.colfilter(df, only=["HI_CHOL", "RIAGENDR", "race", "agecat"])
    python_result = pd.concat([
        clarite.analyze.ewas(phenotype="HI_CHOL", covariates=["agecat", "RIAGENDR"], data=df,
                             survey_design_spec=design),
        clarite.analyze.ewas(phenotype="HI_CHOL", covariates=["race", "RIAGENDR"], data=df,
                             survey_design_spec=design),
        clarite.analyze.ewas(phenotype="HI_CHOL", covariates=["race", "agecat"], data=df,
                             survey_design_spec=design),
        ], axis=0)
    # Compare
    compare_result(r_result, python_result)


def test_nhanes_lonely_certainty():
    """Test the nhanes dataset with a lonely PSU and the value set to certainty"""
    # Load the data
    df = clarite.load.from_csv(DATA_PATH / "nhanes_lonely_data.csv", index_col=None)
    # Load the expected results
    r_result = load_r_results(DATA_PATH / "nhanes_certainty_result.csv")
    # Process data
    df = clarite.modify.make_binary(df, only=["HI_CHOL", "RIAGENDR"])
    df = clarite.modify.make_categorical(df, only=["race", "agecat"])
    design = clarite.survey.SurveyDesignSpec(df, weights="WTMEC2YR", cluster="SDMVPSU", strata="SDMVSTRA",
                                             fpc=None, nest=True, single_cluster='certainty')
    df = clarite.modify.colfilter(df, only=["HI_CHOL", "RIAGENDR", "race", "agecat"])
    python_result = pd.concat([
        clarite.analyze.ewas(phenotype="HI_CHOL", covariates=["agecat", "RIAGENDR"], data=df,
                             survey_design_spec=design),
        clarite.analyze.ewas(phenotype="HI_CHOL", covariates=["race", "RIAGENDR"], data=df,
                             survey_design_spec=design),
        clarite.analyze.ewas(phenotype="HI_CHOL", covariates=["race", "agecat"], data=df,
                             survey_design_spec=design),
        ], axis=0)
    # Compare
    compare_result(r_result, python_result)


def test_nhanes_lonely_adjust():
    """Test the nhanes dataset with a lonely PSU and the value set to adjust"""
    # Load the data
    df = clarite.load.from_csv(DATA_PATH / "nhanes_lonely_data.csv", index_col=None)
    # Load the expected results
    r_result = load_r_results(DATA_PATH / "nhanes_adjust_result.csv")
    # Process data
    df = clarite.modify.make_binary(df, only=["HI_CHOL", "RIAGENDR"])
    df = clarite.modify.make_categorical(df, only=["race", "agecat"])
    design = clarite.survey.SurveyDesignSpec(df, weights="WTMEC2YR", cluster="SDMVPSU", strata="SDMVSTRA",
                                             fpc=None, nest=True, single_cluster='adjust')
    df = clarite.modify.colfilter(df, only=["HI_CHOL", "RIAGENDR", "race", "agecat"])
    python_result = pd.concat([
        clarite.analyze.ewas(phenotype="HI_CHOL", covariates=["agecat", "RIAGENDR"], data=df,
                             survey_design_spec=design),
        clarite.analyze.ewas(phenotype="HI_CHOL", covariates=["race", "RIAGENDR"], data=df,
                             survey_design_spec=design),
        clarite.analyze.ewas(phenotype="HI_CHOL", covariates=["race", "agecat"], data=df,
                             survey_design_spec=design),
        ], axis=0)
    # Compare
    compare_result(r_result, python_result)


def test_nhanes_lonely_average():
    """Test the nhanes dataset with a lonely PSU and the value set to average"""
    # Load the data
    df = clarite.load.from_csv(DATA_PATH / "nhanes_lonely_data.csv", index_col=None)
    # Load the expected results
    r_result = load_r_results(DATA_PATH / "nhanes_average_result.csv")
    # Process data
    df = clarite.modify.make_binary(df, only=["HI_CHOL", "RIAGENDR"])
    df = clarite.modify.make_categorical(df, only=["race", "agecat"])
    design = clarite.survey.SurveyDesignSpec(df, weights="WTMEC2YR", cluster="SDMVPSU", strata="SDMVSTRA",
                                             fpc=None, nest=True, single_cluster='average')
    df = clarite.modify.colfilter(df, only=["HI_CHOL", "RIAGENDR", "race", "agecat"])
    python_result = pd.concat([
        clarite.analyze.ewas(phenotype="HI_CHOL", covariates=["agecat", "RIAGENDR"], data=df,
                             survey_design_spec=design),
        clarite.analyze.ewas(phenotype="HI_CHOL", covariates=["race", "RIAGENDR"], data=df,
                             survey_design_spec=design),
        clarite.analyze.ewas(phenotype="HI_CHOL", covariates=["race", "agecat"], data=df,
                             survey_design_spec=design),
        ], axis=0)
    # Compare
    compare_result(r_result, python_result)


def test_nhanes_realistic():
    """Test a more realistic set of NHANES data, specifically using multiple weights and missing values"""
    # Load the data
    df = clarite.load.from_tsv(DATA_PATH.parent / "test_data_files" / "nhanes_real.txt", index_col="ID")
    # Load the expected results
    r_result = load_r_results(DATA_PATH / "nhanes_real_python.csv")
    # Process data

    # Split out survey info
    survey_cols = ["SDMVPSU", "SDMVSTRA", "WTMEC4YR", "WTSHM4YR", "WTSVOC4Y"]
    survey_df = df[survey_cols]
    df = df.drop(columns=survey_cols)

    df = clarite.modify.make_binary(df, only=["RHQ570", "first_degree_support", "SDDSRVYR",
                                              "female", "black", "mexican",
                                              "other_hispanic", "other_eth"
                                              ])
    df = clarite.modify.make_categorical(df, only=["SES_LEVEL"])
    design = clarite.survey.SurveyDesignSpec(survey_df,
                                             weights={"RHQ570": "WTMEC4YR",
                                                      "first_degree_support": "WTMEC4YR",
                                                      "URXUPT": "WTSHM4YR",
                                                      "LBXV3A": "WTSVOC4Y",
                                                      "LBXBEC": "WTMEC4YR"},
                                             cluster="SDMVPSU",
                                             strata="SDMVSTRA",
                                             fpc=None,
                                             nest=True)
    calculated_result = clarite.analyze.ewas(
        phenotype="BMXBMI",
        covariates=["SES_LEVEL", "SDDSRVYR", "female", "black", "mexican", "other_hispanic", "other_eth", "RIDAGEYR"],
        data=df,
        survey_design_spec=design)
    # Compare
    compare_result(r_result, calculated_result)
