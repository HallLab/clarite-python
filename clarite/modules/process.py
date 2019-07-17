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
    Classify variables into binary, categorical, continuous, and 'check'.  Drop variables that only have NaN values.

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
    result: pd.DataFrame or None
        If inplace, returns None.  Changes the datatypes on the input DataFrame.
        If not inplace, returns a DataFrame with variables that were categorized by setting the datatype
          - binary = 'category' (with 2 categories)
          - categorical = 'category' (with > 2 categories)
          - continuous = numeric (several possible types, usually 'float64' or 'int64')
          - unknown = str


    Examples
    --------
    >>> import clarite
    >>> clarite.process.categorize(nhanes)
    362 of 970 variables (37.32%) are classified as binary (2 unique values).
    47 of 970 variables (4.85%) are classified as categorical (3 to 6 unique values).
    483 of 970 variables (49.79%) are classified as continuous (>= 15 unique values).
    42 of 970 variables (4.33%) were dropped.
            10 variables had zero unique values (all NA).
            32 variables had one unique value.
    36 of 970 variables (3.71%) were not categorized and need to be set manually.
            36 variables had between 6 and 15 unique values
            0 variables had >= 15 values but couldn't be converted to continuous (numeric) values
    """
    # Validate parameters
    assert cat_min > 2
    assert cat_min <= cat_max
    assert cont_min > cat_max

    # Count the number of variables to start with
    total_vars = len(data.columns)
    # Get the number of unique non-na values per variable
    unique_count = data.nunique(dropna=True)

    # No unique non-NA values - Drop these variables
    empty_vars = (unique_count == 0)
    if empty_vars.sum() > 0:
        data = data.drop(columns=empty_vars[empty_vars].index)

    # One unique non-NA value - Drop these variables
    constant_vars = (unique_count == 1)
    if constant_vars.sum() > 0:
        data = data.drop(columns=constant_vars[constant_vars].index)

    # Two unique non-NA values - Convert non-NA values to category (for binary)
    keep_bin = (unique_count == 2)
    if keep_bin.sum() > 0:
        data.loc[:, keep_bin] = data.loc[:, keep_bin].apply(lambda col: col.loc[~col.isna()].astype('category'))

    # Categorical - Convert non-NA values to category type
    keep_cat = (unique_count >= cat_min) & (unique_count <= cat_max)
    if keep_cat.sum() > 0:
        data.loc[:, keep_cat] = data.loc[:, keep_cat].astype('category')  # NaNs are handled correctly, no need skip

    # Continuous - Convert non-NA values to numeric type (even though they probably already are)
    keep_cont = (unique_count >= cont_min)
    check_cont = pd.Series(False, index=keep_cont.index)
    if keep_cont.sum() > 0:
        for col in keep_cont[keep_cont].index:
            try:
                data[col] = pd.to_numeric(data[col])
            except ValueError:
                # Couldn't convert to a number- possibly a categorical variable with string names?
                keep_cont[col] = False
                check_cont[col] = True
                data[col] = data.loc[~col.isna(), col].astype(str)

    # Other - Convert non-NA values to string type
    check_other = ~empty_vars & ~constant_vars & ~keep_bin & ~keep_cat & ~check_cont & ~keep_cont
    if check_other.sum() > 0:
        data.loc[:, check_other] = data.loc[:, check_other].apply(lambda col: col.loc[~col.isna()].astype(str))

    # Log categorized results
    print(f"{keep_bin.sum():,} of {total_vars:,} variables ({keep_bin.sum()/total_vars:.2%}) "
          f"are classified as binary (2 unique values).")
    print(f"{keep_cat.sum():,} of {total_vars:,} variables ({keep_cat.sum()/total_vars:.2%}) "
          f"are classified as categorical ({cat_min} to {cat_max} unique values).")
    print(f"{keep_cont.sum():,} of {total_vars:,} variables ({keep_cont.sum()/total_vars:.2%}) "
          f"are classified as continuous (>= {cont_min} unique values).")

    # Log dropped variables
    dropped = empty_vars.sum() + constant_vars.sum()
    print(f"{dropped:,} of {total_vars:,} variables ({dropped/total_vars:.2%}) were dropped.")
    print(f"\t{empty_vars.sum():,} variables had zero unique values (all NA).")
    print(f"\t{constant_vars.sum():,} variables had one unique value.")

    # Log non-categorized results
    num_not_categorized = check_other.sum() + check_cont.sum()
    print(f"{num_not_categorized:,} of {total_vars:,} variables ({num_not_categorized/total_vars:.2%})"
          f" were not categorized and need to be set manually.")
    print(f"\t{check_other.sum():,} variables had between {cat_max} and {cont_min} unique values")
    print(f"\t{check_cont.sum():,} variables had >= {cont_min} values but couldn't be converted to continuous (numeric) values")

    return data


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
