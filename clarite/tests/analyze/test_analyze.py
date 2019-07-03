from pathlib import Path

import clarite

DATA_PATH = Path(__file__).parent / 'data'


def test_simple_ewas(capfd):
    # Load the data
    df = clarite.io.load_data(DATA_PATH / "nhanes_test.txt", index_col='SEQN', sep="\t")
    out, err = capfd.readouterr()
    assert out == "Loaded 5,932 observations of 42 variables\nA dtypes file was not found, keeping default datatypes\n"
    assert err == ""

    # Separate weight/survey info from the actual variables
    df_survey = df[['WTINT2YR', 'WTMEC2YR', 'SDMVPSU', 'SDMVSTRA', 'WTSH2YR']]
    df = df[[c for c in list(df) if c not in list(df_survey)]]

    # Categorize and recombine
    df_bin, df_cat, df_cont, _ = df.clarite_process.categorize()
    phenotype = "LBDBCDSI"
    covariates = ["RIDAGEYR", "female", "mexican", "other_hisp", "black", "asian", "other_eth"]
    _ = clarite.analyze.ewas(phenotype, covariates, df_bin, df_cat, df_cont)
    out, err = capfd.readouterr()
    assert out == "0 of 37 variables (0.00%) had no non-NA values and are discarded.\n"\
                  "0 of 37 variables (0.00%) had only one value and are discarded.\n"\
                  "29 of 37 variables (78.38%) are classified as binary (2 values).\n"\
                  "1 of 37 variables (2.70%) are classified as categorical (3 to 6 values).\n"\
                  "7 of 37 variables (18.92%) are classified as continuous (>= 15 values).\n"\
                  "0 of 37 variables (0.00%) are not classified (between 6 and 15 values).\n"\
                  "Processed 29 binary variables with 5,932 observations\n"\
                  "Processed 1 categorical variables with 5,932 observations\n"\
                  "Processed 7 continuous variables with 5,932 observations\n"\
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
