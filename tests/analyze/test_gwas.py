import clarite


def test_bams_rec_rec(genotype_case_control):
    result = clarite.analyze.association_study(
        data=genotype_case_control, outcomes="Outcome"
    )
    print()
