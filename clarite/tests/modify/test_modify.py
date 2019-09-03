import re

import pytest
import pandas as pd

from clarite import modify


def test_make_binary(plantTraits, capfd):
    # Fail due to non-binary
    with pytest.raises(ValueError,
                       match=re.escape("11 variable(s) did not have 2 unique values and couldn't be processed "
                                       "as a binary type: pdias, longindex, durflow, height, begflow, mycor, "
                                       "vegaer, vegsout, autopoll, insects, wind")):
        modify.make_binary(plantTraits)

    # Pass, selecting 5 columns known to be binary
    cols = ['piq', 'ros', 'leafy', 'winan', 'suman']
    result = modify.make_binary(plantTraits, only=cols)
    out, err = capfd.readouterr()
    assert out == "================================================================================\n"\
                  "Running make_binary\n"\
                  "--------------------------------------------------------------------------------\n"\
                  "================================================================================\n"\
                  "Running make_binary\n"\
                  "--------------------------------------------------------------------------------\n"\
                  "Set 5 of 31 variable(s) as binary, each with 136 observations\n"\
                  "================================================================================\n"
    assert err == ""
    assert all(result[cols].dtypes == 'category')


def test_make_categorical(plantTraits, capfd):
    """Currently no validation for maximum unique values"""
    result = modify.make_categorical(plantTraits)
    out, err = capfd.readouterr()
    assert out == "================================================================================\n"\
                  "Running make_categorical\n"\
                  "--------------------------------------------------------------------------------\n"\
                  "Set 31 of 31 variable(s) as categorical, each with 136 observations\n"\
                  "================================================================================\n"
    assert err == ""
    assert all(result.dtypes == 'category')


def test_make_continuous(plantTraits, capfd):
    """Currently no validation for minimum unique values"""
    result = modify.make_continuous(plantTraits)
    out, err = capfd.readouterr()
    assert out == "================================================================================\n"\
                  "Running make_continuous\n"\
                  "--------------------------------------------------------------------------------\n"\
                  "Set 31 of 31 variable(s) as continuous, each with 136 observations\n"\
                  "================================================================================\n"
    assert err == ""
    assert all(result.dtypes != 'category')


def test_merge(plantTraits):
    """Merge the different parts of a dataframe and ensure they are merged back to the original"""
    df1 = plantTraits.loc[:, list(plantTraits)[:3]]
    df2 = plantTraits.loc[:, list(plantTraits)[3:6]]
    df3 = plantTraits.loc[:, list(plantTraits)[6:]]
    df = modify.merge_variables(df1, df2)
    df = modify.merge_variables(df, df3)
    assert all(df == plantTraits)


def test_colfilter_percent_zero(plantTraits):
    # TODO
    return


def test_colfilter_min_n(plantTraits):
    # TODO
    return


def test_colfilter_min_cat_n(plantTraits):
    # TODO
    return


def test_rowfilter_incomplete_obs(plantTraits):
    # TODO
    return


def test_recode_values(plantTraits):
    # TODO
    return


def test_remove_outliers(plantTraits):
    # TODO
    return


def test_categorize(plantTraits, capfd):
    # TODO
    modify.categorize(plantTraits)
    return


def test_transform(plantTraits, capfd):
    """Test a log10 transform"""
    df = pd.DataFrame({
        'a': [10, 100, 1000],
        'b': [100, 1000, 10000],
        'c': [True, False, True]
    })

    result = modify.transform(df, 'log10', skip=['c'])

    assert all(result['a'] == [1, 2, 3])
    assert all(result['b'] == [2, 3, 4])
    assert all(result['c'] == [True, False, True])
    return
