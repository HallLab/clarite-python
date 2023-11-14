# import numpy as np
# import pandas as pd
import pytest

import clarite

# from clarite.modules.survey import SurveyDesignSpec


def test_bams_main(genotype_case_control_add_add_main):
    result = clarite.analyze.association_study(
        data=genotype_case_control_add_add_main, outcomes="Outcome"
    )
    assert result.loc[("SNP1", "Outcome"), "pvalue"] <= 1e-5
    assert result.loc[("SNP2", "Outcome"), "pvalue"] <= 1e-5


def test_bams_interaction(genotype_case_control_rec_rec_onlyinteraction):
    # Based on the model, neither SNP should be very significant in an association study
    result = clarite.analyze.association_study(
        data=genotype_case_control_rec_rec_onlyinteraction, outcomes="Outcome"
    )
    assert result.loc[("SNP1", "Outcome"), "pvalue"] > 1e-5
    assert result.loc[("SNP2", "Outcome"), "pvalue"] > 1e-5

    # Significant in an interaction test
    result_interaction = clarite.analyze.interaction_study(
        data=genotype_case_control_rec_rec_onlyinteraction, outcomes="Outcome"
    )
    assert result_interaction.loc[("SNP1", "SNP2", "Outcome"), "LRT_pvalue"] <= 1e-5


# @pytest.mark.slow
# @pytest.mark.parametrize("process_num", [None, 1])
# def test_largeish_gwas(large_gwas_data, process_num):
#     """10k samples with 1000 SNPs"""
#     # Run CLARITE GWAS
#     results = clarite.analyze.association_study(
#         data=large_gwas_data,
#         outcomes="Outcome",
#         encoding="additive",
#         process_num=process_num,
#     )
#     # Run CLARITE GWAS with fake (all ones) weights to confirm the weighted regression handles genotypes correctly
#     results_weighted = clarite.analyze.association_study(
#         data=large_gwas_data,
#         outcomes="Outcome",
#         encoding="additive",
#         process_num=process_num,
#         survey_design_spec=SurveyDesignSpec(
#             survey_df=pd.DataFrame({"weights": np.ones(len(large_gwas_data))}),
#             weights="weights",
#         ),
#     )
#     assert results == results
#     assert results_weighted == results_weighted
#     # TODO: Add useful asserts rather than just making sure it runs


@pytest.mark.xfail(strict=True)
def test_gwas_r(large_gwas_data):
    clarite.analyze.association_study(
        data=large_gwas_data,
        outcomes="Outcome",
        encoding="additive",
        survey_kind="r_survey",
    )
