"""
Process
========

Functions used to process data from one form into another, such as categorizing variables and placing them in separate DataFrames

  .. autosummary::
     :toctree: modules/process

     categorize
     merge_variables
     move_variables

"""

from typing import List, Optional, Union

import pandas as pd

from ..internal.utilities import _validate_skip_only


def categorize(data: pd.DataFrame, cat_min: int = 3, cat_max: int = 6, cont_min: int = 15):
    """
    Divide variables into binary, categorical, continuous, and ambiguous dataframes

    Parameters
    ----------
    data: pd.DataFrame
        The DataFrame to be processed
    cat_min: int, default 3
        Minimum number of unique, non-NA values for a categorical variable
    cat_max: int, default 6
        Maximum number of unique, non-NA values for a categorical variable
    cont_min: int, default 15
        Minimum number of unique, non-NA values for a continuous variable

    Returns
    -------
    bin_df: pd.DataFrame
        DataFrame with variables that were categorized as *binary*
    cat_df: pd.DataFrame
        DataFrame with variables that were categorized as *categorical*
    bin_df: pd.DataFrame
        DataFrame with variables that were categorized as *continuous*
    other_df: pd.DataFrame
        DataFrame with variables that were not categorized and should be examined manually

    Examples
    --------
    >>> import clarite
    >>> nhanes_bin, nhanes_cat, nhanes_cont, nhanes_other = clarite.process.categorize()
    10 of 945 variables (1.06%) had no non-NA values and are discarded.
    33 of 945 variables (3.49%) had only one value and are discarded.
    361 of 945 variables (38.20%) are classified as binary (2 values).
    44 of 945 variables (4.66%) are classified as categorical (3 to 6 values).
    461 of 945 variables (48.78%) are classified as continuous (>= 15 values).
    36 of 945 variables (3.81%) are not classified (between 6 and 15 values).
    """
    # Validate parameters
    assert cat_min > 2
    assert cat_min <= cat_max
    assert cont_min > cat_max

    # Create filter series
    num_before = len(data.columns)
    unique_count = data.nunique()

    # No values (All NA)
    zero_filter = unique_count == 0
    num_zero = sum(zero_filter)
    print(f"{num_zero:,} of {num_before:,} variables ({num_zero/num_before:.2%}) had no non-NA values and are discarded.")

    # Single value variables (useless for regression)
    single_filter = unique_count == 1
    num_single = sum(single_filter)
    print(f"{num_single:,} of {num_before:,} variables ({num_single/num_before:.2%}) had only one value and are discarded.")

    # Binary
    binary_filter = unique_count == 2
    num_binary = sum(binary_filter)
    print(f"{num_binary:,} of {num_before:,} variables ({num_binary/num_before:.2%}) are classified as binary (2 values).")
    bin_df = data.loc[:, binary_filter]

    # Categorical
    cat_filter = (unique_count >= cat_min) & (unique_count <= cat_max)
    num_cat = sum(cat_filter)
    print(f"{num_cat:,} of {num_before:,} variables ({num_cat/num_before:.2%}) are classified as categorical ({cat_min} to {cat_max} values).")
    cat_df = data.loc[:, cat_filter]

    # Continuous
    cont_filter = unique_count >= cont_min
    num_cont = sum(cont_filter)
    print(f"{num_cont:,} of {num_before:,} variables ({num_cont/num_before:.2%}) are classified as continuous (>= {cont_min} values).")
    cont_df = data.loc[:, cont_filter]

    # Other
    other_filter = ~zero_filter & ~single_filter & ~binary_filter & ~cat_filter & ~cont_filter
    num_other = sum(other_filter)
    print(f"{num_other:,} of {num_before:,} variables ({num_other/num_before:.2%}) are not classified (between {cat_max} and {cont_min} values).")
    other_df = data.loc[:, other_filter]

    return bin_df, cat_df, cont_df, other_df


def merge_variables(left: pd.DataFrame, right: pd.DataFrame, how: str = 'outer'):
    """
    Merge a list of dataframes with different variables side-by-side.  Keep all observations ('outer' merge) by default.

    Parameters
    ----------
    left: pd.Dataframe
        "left" DataFrame
    right: pd.DataFrame
        "right" DataFrame which uses the same index
    how: merge method, one of {'left', 'right', 'inner', 'outer'}
        Keep only rows present in the left data, the right data, both datasets, or either dataset.

    Examples
    --------
    >>> import clarite
    >>> df = clarite.modify.merge_variables(df_bin, df_cat, how='outer')
    """
    return left.merge(right, left_index=True, right_index=True, how=how)


def move_variables(left: pd.DataFrame, right: pd.DataFrame,
                   skip: Optional[Union[str, List[str]]] = None, only: Optional[Union[str, List[str]]] = None):
    """
    Move one or more variables from one DataFrame to another

    Parameters
    ----------
    left: pd.Dataframe
        DataFrame containing the variable(s) to be moved
    right: pd.DataFrame
        DataFrame (which uses the same index) that the variable(s) will be moved to
    skip: str, list or None (default is None)
        List of variables that will *not* be moved
    only: str, list or None (default is None)
        List of variables that are the *only* ones to be moved

    Returns
    -------
    left: pd.DataFrame
        The first DataFrame with the variables removed
    right: pd.DataFrame
        The second DataFrame with the variables added

    Examples
    --------
    >>> import clarite
    >>> df_cat, df_cont = clarity.process.move_variables(df_cat, df_cont, only=["DRD350AQ", "DRD350DQ", "DRD350GQ"])
    Moved 3 variables.
    >>> discovery_check, discovery_cont = clarite.process.move_variables(discovery_check, discovery_cont)
    Moved 39 variables.
    """
    # Which columns
    columns = _validate_skip_only(list(left), skip, only)

    # Add to new df
    right = merge_variables(right, left[columns])

    # Remove from original
    left = left.drop(columns, axis='columns')

    # Log
    if len(columns) == 1:
        print("Moved 1 variable.")
    else:
        print(f"Moved {len(columns)} variables.")

    # Return
    return left, right
