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
from collections import Counter
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
    categorized_df: pd.DataFrame
        DataFrame with variables that were categorized by setting the datatype
          - binary = 'bool'
          - categorical = 'category'
          - continuous = numeric (several possible types)
          - unknown = str


    Examples
    --------
    >>> import clarite
    >>> nhanes = clarite.process.categorize()
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
    unique_count = data.nunique(dropna=True)

    # Count classifications
    counts = Counter()

    # Process each column
    for col in data.columns:
        if unique_count[col] == 0:
            data[col] = data[col].astype(str)
            counts['check_zero'] += 1
        elif unique_count[col] == 1:
            data[col] = data[col].astype(str)
            counts['check_one'] += 1
        elif unique_count[col] == 2:
            if set(data[col].unique()) == {0, 1}:
                data[col] = data[col].astype(bool)
                counts['keep_binary'] += 1
            else:
                data[col] = data[col].astype(str)
                counts['check_binary'] += 1
        elif (unique_count[col] >= cat_min) & (unique_count[col] <= cat_max):
            data[col] = data[col].astype('category')
            counts['keep_categorical'] += 1
        elif (unique_count[col] >= cont_min):
            data[col] = pd.to_numeric(data[col])
            counts['keep_continuous'] += 1
        else:
            data[col] = data[col].astype(str)
            counts['check_other'] += 1

    # Log results
    num_binary = counts['keep_binary']
    num_cat = counts['keep_categorical']
    num_cont = counts['keep_continuous']
    num_categorized = num_binary + num_cat + num_cont
    print(f"{num_categorized:,} of {num_before:,} were categorized:")
    print(f"\t{num_binary:,} of {num_before:,} variables ({num_binary/num_before:.2%}) are classified as binary (2 values).")
    print(f"\t{num_cat:,} of {num_before:,} variables ({num_cat/num_before:.2%}) are classified as categorical ({cat_min} to {cat_max} values).")
    print(f"\t{num_cont:,} of {num_before:,} variables ({num_cont/num_before:.2%}) are classified as continuous (>= {cont_min} values).")
    print(f"{num_before - num_categorized} of {num_before:,} were not categorized.")
    print(f"\t{counts['check_zero']:,} of {num_before:,} variables ({counts['check_zero']/num_before:.2%}) had zero values (all NA)")
    print(f"\t{counts['check_one']:,} of {num_before:,} variables ({counts['check_one']/num_before:.2%}) had one unique value")
    print(f"\t{counts['check_binary']:,} of {num_before:,} variables ({counts['check_binary']/num_before:.2%}) had two unique values, but were not coded as '0' and '1'")
    print(f"\t{counts['check_other']:,} of {num_before:,} variables ({counts['check_other']/num_before:.2%}) had between {cat_max} and {cont_min} unique values")

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
