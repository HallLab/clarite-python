import pytest

from clarite import utilities


def test_make_binary(plantTraits, capfd):
    # Fail due to non-binary
    with pytest.raises(ValueError, match="11 of 31 variables did not have 2 unique values and couldn't be processed as a binary type"):
        utilities.make_binary(plantTraits)

    # Pass, selecting 5 columns known to be binary
    result = utilities.make_binary(plantTraits[['piq', 'ros', 'leafy', 'winan', 'suman']])
    out, err = capfd.readouterr()
    assert out == "Processed 5 binary variables with 136 observations\n"
    assert err == ""
    assert all(result.dtypes == 'category')


def test_make_categorical(plantTraits, capfd):
    """Currently no validation for maximum unique values"""
    result = utilities.make_categorical(plantTraits)
    out, err = capfd.readouterr()
    assert out == "Processed 31 categorical variables with 136 observations\n"
    assert err == ""
    assert all(result.dtypes == 'category')


def test_make_continuous(plantTraits, capfd):
    """Currently no validation for minimum unique values"""
    result = utilities.make_continuous(plantTraits)
    out, err = capfd.readouterr()
    assert out == "Processed 31 continuous variables with 136 observations\n"
    assert err == ""
    assert all(result.dtypes != 'category')


def test_merge(plantTraits):
    """Merge the different parts of a dataframe and ensure they are merged back to the original"""
    df1 = plantTraits.loc[:, list(plantTraits)[:3]]
    df2 = plantTraits.loc[:, list(plantTraits)[3:6]]
    df3 = plantTraits.loc[:, list(plantTraits)[6:]]
    df = utilities.merge_variables([df1, df2, df3])
    assert all(df == plantTraits)
