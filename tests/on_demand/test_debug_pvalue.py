from pathlib import Path

import clarite

# import numpy as np
# import pandas as pd


def test_interactions_debug():
    v_path_current = Path(__file__).parent
    path_nhames_disc = str(v_path_current) + "/Analysis_2_discovery_all_bmi.txt"
    df_nhames_disc_2 = clarite.load.from_tsv(path_nhames_disc)

    # Setup columns rules
    # list of Outcomes
    list_outcome = ["logBMI"]

    # list of Covariants
    list_covariant = [
        "female",
        "black",
        "mexican",
        "other_hispanic",
        "other_eth",
        "SDDSRVYR",
        "SES_LEVEL",
        "RIDAGEYR",
    ]
    # conver_pvalue = ["SMQ020", "DR1TCAFF"]
    # conver_sem_pvalue = ["LBXPFDE", "LBXWCF"]
    # no_conver = []

    e1 = "SMQ020"
    e2 = "DR1TCAFF"

    # Keeo only columns to run regression
    df_interactions = df_nhames_disc_2.loc[
        :, list_covariant + list_outcome + list([e1, e2])
    ]
    # TODO: we need treat NaN?
    # df_interactions = df_interactions.fillna(0)
    df_inter = clarite.analyze.interaction_study(
        data=df_interactions,
        outcomes=list_outcome,
        interactions=[(e1, e2)],
        covariates=list_covariant,
        report_betas=True,
    )

    print(df_inter)
    assert 2 == 2
