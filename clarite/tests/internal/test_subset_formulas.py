from clarite.internal.formulas import SubsetFormula


def test_subset_parse():
    # TODO - add more tests
    SubsetFormula("x > 4")
    SubsetFormula("x==4")
    return
