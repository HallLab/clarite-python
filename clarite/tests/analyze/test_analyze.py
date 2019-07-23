from pathlib import Path

import clarite

DATA_PATH = Path(__file__).parent / 'data'


def test_simple_ewas(capfd):
    # Load the data
    df = clarite.load.from_tsv(DATA_PATH / "nhanes_test.txt", index_col='SEQN')
    out, err = capfd.readouterr()
    assert out == "Loaded 5,932 observations of 42 variables\n"
    assert err == ""

    # Separate weight/survey info from the actual variables
    df_survey = df[['WTINT2YR', 'WTMEC2YR', 'SDMVPSU', 'SDMVSTRA', 'WTSH2YR']]
    df = df[[c for c in list(df) if c not in list(df_survey)]]

    # Categorize and recombine
    data = clarite.modify.categorize(df)
    phenotype = "LBDBCDSI"
    covariates = ["RIDAGEYR", "female", "mexican", "other_hisp", "black", "asian", "other_eth"]
    _ = clarite.analyze.ewas(phenotype, covariates, data)
    out, err = capfd.readouterr()
    assert out == "================================================================================\n"\
                  "Running categorize\n"\
                  "--------------------------------------------------------------------------------\n"\
                  "29 of 37 variables (78.38%) are classified as binary (2 unique values).\n"\
                  "1 of 37 variables (2.70%) are classified as categorical (3 to 6 unique values).\n"\
                  "7 of 37 variables (18.92%) are classified as continuous (>= 15 unique values).\n"\
                  "0 of 37 variables (0.00%) were dropped.\n"\
                  "0 of 37 variables (0.00%) were not categorized and need to be set manually.\n"\
                  "================================================================================\n"\
                  "Running EWAS on a continuous variable\n"\
                  "\n"\
                  "####### Regressing 5 Continuous Variables #######\n"\
                  "\n"\
                  "\n"\
                  "####### Regressing 23 Binary Variables #######\n"\
                  "\n"\
                  "MCQ170K = NULL due to: too few complete obervations (153 < 200)\n"\
                  "MCQ170L = NULL due to: too few complete obervations (104 < 200)\n"\
                  "\n"\
                  "####### Regressing 1 Categorical Variables #######\n"\
                  "\n"\
                  "Completed EWAS\n"\
                  "\n"
    assert err == ""
