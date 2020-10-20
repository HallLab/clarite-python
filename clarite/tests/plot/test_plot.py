from pathlib import Path
import pandas as pd
import pytest

import clarite

DATA_PATH = Path(__file__).parent.parent / 'test_data_files'
PY_DATA_PATH = Path(__file__).parent.parent / 'py_test_output'


@pytest.fixture
def resultNHANESReal():
    # Load the data
    df = clarite.load.from_tsv(DATA_PATH / "nhanes_real.txt", index_col="ID")

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
    return calculated_result


@pytest.fixture
def resultNHANESsmall():
    # Load the data
    df = clarite.load.from_csv(DATA_PATH / "nhanes_data.csv", index_col=None)
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
    clarite.analyze.add_corrected_pvalues(python_result)
    return python_result


def test_top_results_nhanesreal(resultNHANESReal, capfd):
    clarite.plot.top_results(resultNHANESReal, "pvalue", cutoff=0.05, num_rows=3,
                             filename=PY_DATA_PATH/"top_results_nhanesreal.png")


def test_top_results_nhanesreal_no_cutoff(resultNHANESReal, capfd):
    clarite.plot.top_results(resultNHANESReal, "pvalue", cutoff=None, num_rows=3,
                             filename=PY_DATA_PATH/"top_results_nhanesreal_no_cutoff.png")


def test_top_results_nhanessmall(resultNHANESsmall, capfd):
    clarite.plot.top_results(resultNHANESsmall, "pvalue_bonferroni", cutoff=0.05,
                             filename=PY_DATA_PATH / "top_results_nhanessmall.png")


def test_top_results_multiphenotype(resultNHANESsmall, capfd):
    data = resultNHANESsmall.copy().reset_index()
    data.loc[0, "Phenotype"] = "Other"
    data.set_index(["Variable", "Phenotype"])
    with pytest.raises(ValueError):
        clarite.plot.top_results(data,
                                 "pvalue_bonferroni",
                                 cutoff=0.05,
                                 filename=PY_DATA_PATH / "top_results_multiphenotype.png")
