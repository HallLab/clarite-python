from pathlib import Path
import pandas as pd
import pytest

import clarite

DATA_PATH = Path(__file__).parent.parent / "test_data_files"
PY_DATA_PATH = Path(__file__).parent.parent / "py_test_output"


@pytest.fixture
def resultNHANESReal():
    # Load the data
    df = clarite.load.from_tsv(DATA_PATH / "nhanes_real.txt", index_col="ID")

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
    calculated_result = clarite.analyze.ewas(
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
    clarite.analyze.add_corrected_pvalues(calculated_result)
    return calculated_result


@pytest.fixture
def resultNHANESReal_multi(resultNHANESReal):
    top = resultNHANESReal.copy()
    bottom = resultNHANESReal.reset_index(drop=False)
    bottom["Outcome"] = "AlsoBMXBMI"
    bottom = bottom.set_index(resultNHANESReal.index.names)
    bottom["pvalue"] = bottom["pvalue"] / 10
    result = pd.concat([top, bottom])
    clarite.analyze.add_corrected_pvalues(result)
    return result


@pytest.fixture
def resultNHANESsmall():
    # Load the data
    df = clarite.load.from_csv(DATA_PATH / "nhanes_data.csv", index_col=None)
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
    )
    df = clarite.modify.colfilter(df, only=["HI_CHOL", "RIAGENDR", "race", "agecat"])
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
    clarite.analyze.add_corrected_pvalues(python_result)
    return python_result


@pytest.mark.parametrize(
    "ewas_result_list,bonferroni,fdr,label_vars",
    [
        (["resultNHANESReal"], None, None, None),
        (["resultNHANESReal", "resultNHANESsmall"], None, None, None),
        (["resultNHANESReal", "resultNHANESsmall"], 0.05, 0.1, ["LBXBEC"]),
        (["resultNHANESReal_multi"], None, None, ["LBXBEC"]),
    ],
)
def test_manhattan(ewas_result_list, bonferroni, fdr, label_vars, request):
    dfs = {name: request.getfixturevalue(name) for name in ewas_result_list}
    clarite.plot.manhattan(
        dfs=dfs, bonferroni=bonferroni, fdr=fdr, label_vars=label_vars
    )


@pytest.mark.parametrize(
    "ewas_result_name,pvalue_name,cutoff,num_rows,filename",
    [
        (
            "resultNHANESReal",
            "pvalue",
            0.05,
            3,
            PY_DATA_PATH / "top_results_nhanesreal.png",
        ),
        (
            "resultNHANESReal",
            "pvalue",
            None,
            3,
            PY_DATA_PATH / "top_results_nhanesreal_no_cutoff.png",
        ),
        (
            "resultNHANESsmall",
            "pvalue_bonferroni",
            0.05,
            None,
            PY_DATA_PATH / "top_results_nhanessmall.png",
        ),
        pytest.param(
            "resultNHANESsmall",
            "pvalue_bonferroni",
            0.05,
            None,
            PY_DATA_PATH / "top_results_multioutcome.png",
            marks=pytest.mark.xfail,
        ),
    ],
)
def test_top_results(
    ewas_result_name, pvalue_name, cutoff, num_rows, filename, request
):
    ewas_result = request.getfixturevalue(ewas_result_name)
    clarite.plot.top_results(
        ewas_result=ewas_result,
        pvalue_name=pvalue_name,
        cutoff=cutoff,
        num_rows=num_rows,
        filename=filename,
    )
