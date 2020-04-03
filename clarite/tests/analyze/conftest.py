def compare_result(loaded_result, calculated_result):
    """Binary variables must be specified, since there are expected differences"""
    merged = pd.merge(left=loaded_result, right=calculated_result,
                      left_index=True, right_index=True,
                      how="inner", suffixes=("_loaded", "_calculated"))
    try:
        assert len(merged) == len(loaded_result) == len(calculated_result)
    except AssertionError:
        raise ValueError(f" Loaded Results have {len(loaded_result):,} rows,"
                         f" Calculated results have {len(calculated_result):,} rows,"
                         f" merged data has {len(merged):,} rows")
    # Close-enough equality of numeric values
    for var in ["N", "Beta", "SE", "Variable_pvalue", "LRT_pvalue", "pvalue"]:
        try:
            assert np.allclose(merged[f"{var}_loaded"], merged[f"{var}_calculated"], equal_nan=True, atol=0)
        except AssertionError:
            raise ValueError(f"{var}:\n"
                             f"{merged[f'{var}_loaded']}\n"
                             f"{merged[f'{var}_calculated']}")
    for var in ["Diff_AIC"]:
        # Pass if R result is NaN (quasibinomial) or Python result is NaN (survey data used)
        either_nan = merged[[f'{var}_loaded', f'{var}_calculated']].isna().any(axis=1)
        try:
            # Value must be close when both exist or both are NaN
            assert np.allclose(merged.loc[~either_nan, f"{var}_loaded"],
                               merged.loc[~either_nan, f"{var}_calculated"], equal_nan=True)
        except AssertionError:
            raise ValueError(f"{var}: Loaded ({merged[f'{var}_r']}) != Calculated ({merged[f'{var}_python']})")

    # Both converged
    assert all(merged["Converged_loaded"] == merged["Converged_calculated"])