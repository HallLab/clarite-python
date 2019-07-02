import clarite


def test_categorize(plantTraits, capfd):
    df_bin, df_cat, df_cont, df_ambiguous = plantTraits.clarite_process.categorize()
    out, err = capfd.readouterr()
    assert out == "0 of 31 variables (0.00%) had no non-NA values and are discarded.\n"\
                  "0 of 31 variables (0.00%) had only one value and are discarded.\n"\
                  "20 of 31 variables (64.52%) are classified as binary (2 values).\n"\
                  "6 of 31 variables (19.35%) are classified as categorical (3 to 6 values).\n"\
                  "2 of 31 variables (6.45%) are classified as continuous (>= 15 values).\n"\
                  "3 of 31 variables (9.68%) are not classified (between 6 and 15 values).\n"
    assert err == ""

    assert all(df_bin.nunique() == 2)
    assert df_bin.shape == (136, 20)
    clarite.modify.make_binary(df_bin)

    assert all((df_cat.nunique() >= 3) & (df_cat.nunique() <= 6))
    assert df_cat.shape == (136, 6)
    clarite.modify.make_categorical(df_cat)

    assert all(df_cont.nunique() >= 15)
    assert df_cont.shape == (136, 2)
    clarite.modify.make_continuous(df_cont)

    assert all((df_ambiguous.nunique() > 6) & (df_ambiguous.nunique() < 15))
    assert df_ambiguous.shape == (136, 3)
