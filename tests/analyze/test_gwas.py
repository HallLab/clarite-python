import clarite


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
