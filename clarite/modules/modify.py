"""
Modify
======

Functions used to filter and/or change some data, always taking in one set of data and returning one set of data.

  **CLI Command**: ``modify``

  .. autosummary::
     :toctree: modules/modify

     colfilter_percent_zero
     colfilter_min_n
     colfilter_min_cat_n
     rowfilter_incomplete_obs
     recode_values
     remove_outliers
     make_binary
     make_categorical
     make_continuous

"""

from typing import Optional, List, Union

import numpy as np
import pandas as pd

from ..internal.utilities import _validate_skip_only


def colfilter_percent_zero(data: pd.DataFrame, filter_percent: float = 90.0,
                           skip: Optional[Union[str, List[str]]] = None, only: Optional[Union[str, List[str]]] = None):
    """
    Remove columns which have <proportion> or more values of zero (excluding NA)

    Parameters
    ----------
    data: pd.DataFrame
        The DataFrame to be processed and returned
    filter_percent: float, default 90.0
            If the percentage of rows in the data with a value of zero is greater than or equal to this value, the variable is filtered out.
    skip: str, list or None (default is None)
        List of variables that the filter should *not* be applied to
    only: str, list or None (default is None)
        List of variables that the filter should *only* be applied to

    Returns
    -------
    data: pd.DataFrame
        The filtered DataFrame

    Examples
    --------
    >>> import clarite
    >>> nhanes_discovery_cont = clarite.modify.colfilter_percent_zero(nhanes_discovery_cont)
    Removed 30 of 369 variables (8.13%) which were equal to zero in at least 90.00% of non-NA observations.
    """
    columns = _validate_skip_only(list(data), skip, only)
    num_before = len(data.columns)

    percent_value = 100 * data.apply(lambda col: sum(col == 0) / col.count())
    kept = (percent_value < filter_percent) | ~data.columns.isin(columns)
    num_removed = num_before - sum(kept)

    print(f"Removed {num_removed:,} of {num_before:,} variables ({num_removed/num_before:.2%}) "
          f"which were equal to zero in at least {filter_percent:.2f}% of non-NA observations.")
    return data.loc[:, kept]


def colfilter_min_n(data: pd.DataFrame, n: int = 200,
                    skip: Optional[Union[str, List[str]]] = None, only: Optional[Union[str, List[str]]] = None):
    """
    Remove columns which have less than <n> unique values (excluding NA)

    Parameters
    ----------
    data: pd.DataFrame
        The DataFrame to be processed and returned
    n: int, default 200
        The minimum number of unique values required in order for a variable not to be filtered
    skip: str, list or None (default is None)
        List of variables that the filter should *not* be applied to
    only: str, list or None (default is None)
        List of variables that the filter should *only* be applied to

    Returns
    -------
    data: pd.DataFrame
        The filtered DataFrame

    Examples
    --------
    >>> import clarite
    >>> nhanes_discovery_bin = clarite.modify.colfilter_min_n(nhanes_discovery_bin)
    Removed 129 of 361 variables (35.73%) which had less than 200 values
    """
    columns = _validate_skip_only(list(data), skip, only)
    num_before = len(data.columns)

    counts = data.count()  # by default axis=0 (rows) so counts number of non-NA rows in each column
    kept = (counts >= n) | ~data.columns.isin(columns)
    num_removed = num_before - sum(kept)

    print(f"Removed {num_removed:,} of {num_before:,} variables ({num_removed/num_before:.2%}) which had less than {n} values")
    return data.loc[:, kept]


def colfilter_min_cat_n(data, n: int = 200, skip: Optional[Union[str, List[str]]] = None, only: Optional[Union[str, List[str]]] = None):
    """
    Remove columns which have less than <n> occurences of each unique value

    Parameters
    ----------
    data: pd.DataFrame
        The DataFrame to be processed and returned
    n: int, default 200
        The minimum number of occurences of each unique value required in order for a variable not to be filtered
    skip: str, list or None (default is None)
        List of variables that the filter should *not* be applied to
    only: str, list or None (default is None)
        List of variables that the filter should *only* be applied to

    Returns
    -------
    data: pd.DataFrame
        The filtered DataFrame

    Examples
    --------
    >>> import clarite
    >>> nhanes_discovery_bin = clarite.modify.colfilter_min_cat_n(nhanes_discovery_bin)
    Removed 159 of 232 variables (68.53%) which had a category with less than 200 values
    """
    columns = _validate_skip_only(list(data), skip, only)
    num_before = len(data.columns)

    min_category_counts = data.apply(lambda col: col.value_counts().min())
    kept = (min_category_counts >= n) | ~data.columns.isin(columns)
    num_removed = num_before - sum(kept)

    print(f"Removed {num_removed:,} of {num_before:,} variables ({num_removed/num_before:.2%}) which had a category with less than {n} values")
    return data.loc[:, kept]


def rowfilter_incomplete_obs(data, skip: Optional[Union[str, List[str]]] = None, only: Optional[Union[str, List[str]]] = None):
    """
    Remove rows containing null values

    Parameters
    ----------
    data: pd.DataFrame
        The DataFrame to be processed and returned
    skip: str, list or None (default is None)
        List of columns that are not checked for null values
    only: str, list or None (default is None)
        List of columns that are the only ones to be checked for null values

    Returns
    -------
    data: pd.DataFrame
        The filtered DataFrame

    Examples
    --------
    >>> import clarite
    >>> nhanes = clarite.modify.rowfilter_incomplete_obs(only=[phenotype] + covariates)
    Removed 3,687 of 22,624 rows (16.30%) due to NA values in the specified columns
    """
    columns = _validate_skip_only(list(data), skip, only)

    keep_IDs = data[columns].isnull().sum(axis=1) == 0  # Number of NA in each row is 0
    n_removed = len(data) - sum(keep_IDs)

    print(f"Removed {n_removed:,} of {len(data):,} rows ({n_removed/len(data):.2%}) due to NA values in the specified columns")
    return data[keep_IDs]


def recode_values(data, replacement_dict,
                  skip: Optional[Union[str, List[str]]] = None, only: Optional[Union[str, List[str]]] = None):
    """
    Convert values in a dataframe.  By default, replacement occurs in all columns but this may be modified with 'skip' or 'only'.
    Pandas has more powerful 'replace' methods for more complicated scenarios.

    Parameters
    ----------
    data: pd.DataFrame
        The DataFrame to be processed and returned
    replacement_dict: dictionary
        A dictionary mapping the value being replaced to the value being inserted
    skip: str, list or None (default is None)
        List of variables that the replacement should *not* be applied to
    only: str, list or None (default is None)
        List of variables that the replacement should *only* be applied to

    Examples
    --------
    >>> import clarite
    >>> clarite.recode_values(df, {7: np.nan, 9: np.nan}, only=['SMQ077', 'DBD100'])
    Replaced 17 values from 22,624 rows in 2 columns
    >>> clarite.recode_values(df, {10: 12}, only=['SMQ077', 'DBD100'])
    No occurences of replaceable values were found, so nothing was replaced.
    """
    # Limit columns if needed
    if skip is not None or only is not None:
        columns = _validate_skip_only(list(data), skip, only)
        replacement_dict = {c: replacement_dict for c in columns}

    # Replace
    result = data.replace(to_replace=replacement_dict, value=None, inplace=False)

    # Log
    diff = result.eq(data)
    diff[pd.isnull(result) == pd.isnull(data)] = True  # NAs are not equal by default
    diff = ~diff  # not True where a value was replaced
    cols_with_changes = (diff.sum() > 0).sum()
    cells_with_changes = diff.sum().sum()
    if cells_with_changes > 0:
        print(f"\tReplaced {cells_with_changes:,} values from {len(data):,} rows in {cols_with_changes:,} columns")
    else:
        print(f"\tNo occurences of replaceable values were found, so nothing was replaced.")

    # Return
    return result


def remove_outliers(data, method: str = 'gaussian', cutoff=3,
                    skip: Optional[Union[str, List[str]]] = None, only: Optional[Union[str, List[str]]] = None):
    """
    Remove outliers from the dataframe by replacing them with np.nan

    Parameters
    ----------
    data: pd.DataFrame
        The DataFrame to be processed and returned
    method: string, 'gaussian' (default) or 'iqr'
        Define outliers using a gaussian approach (standard deviations from the mean) or inter-quartile range
    cutoff: positive numeric, default of 3
        Either the number of standard deviations from the mean (method='gaussian') or the multiple of the IQR (method='iqr')
        Any values equal to or more extreme will be replaced with np.nan
    skip: str, list or None (default is None)
        List of variables that the replacement should *not* be applied to
    only: str, list or None (default is None)
        List of variables that the replacement should *only* be applied to

    Examples
    --------
    >>> import clarite
    >>> nhanes_rm_outliers = clarite.remove_outliers(nhanes, method='iqr', cutoff=1.5, only=['DR1TVB1', 'URXP07', 'SMQ077'])
    Removing outliers with values < 1st Quartile - (1.5 * IQR) or > 3rd quartile + (1.5 * IQR) in 3 columns
        430 of 22,624 rows of URXP07 were outliers
        730 of 22,624 rows of DR1TVB1 were outliers
        Skipped filtering 'SMQ077' because it is a categorical variable
    >>> nhanes_rm_outliers = clarite.remove_outliers(only=['DR1TVB1', 'URXP07'])
    Removing outliers with values more than 3 standard deviations from the mean in 2 columns
        42 of 22,624 rows of URXP07 were outliers
        301 of 22,624 rows of DR1TVB1 were outliers
    """
    # Copy to avoid replacing in-place
    data = data.copy(deep=True)

    # Which columns
    columns = _validate_skip_only(list(data), skip, only)

    # Check cutoff and method, printing what is being done
    if cutoff <= 0:
        raise ValueError("'cutoff' must be >= 0")
    if method == 'iqr':
        print(f"Removing outliers with values < 1st Quartile - ({cutoff} * IQR) or > 3rd quartile + ({cutoff} * IQR) in {len(columns):,} columns")
    elif method == 'gaussian':
        print(f"Removing outliers with values more than {cutoff} standard deviations from the mean in {len(columns):,} columns")
    else:
        raise ValueError(f"'{method}' is not a supported method for outlier removal - only 'gaussian' and 'iqr'.")

    # Process each column
    for c in columns:
        if str(data.dtypes[c]) == 'category':
            print(f"\tSkipped filtering '{c}' because it is a categorical variable")
            continue
        # Calculate outliers
        if method == 'iqr':
            q1 = data[c].quantile(0.25)
            q3 = data[c].quantile(0.75)
            iqr = abs(q3 - q1)
            bottom = q1 - (iqr * cutoff)
            top = q3 + (iqr * cutoff)
            outliers = (data[c] < bottom) | (data[c] > top)
        elif method == 'gaussian':
            outliers = (data[c]-data[c].mean()).abs() > cutoff*data[c].std()
        # If there are any, log a message and replace them with np.nan
        if sum(outliers) > 0:
            print(f"\t{sum(outliers):,} of {len(data):,} rows of {c} were outliers")
            data.loc[outliers, c] = np.nan

    return data


def make_binary(data: pd.DataFrame, skip: Optional[Union[str, List[str]]] = None, only: Optional[Union[str, List[str]]] = None):
    """
    Validate and type a dataframe of binary variables

    Checks that each variable has at most 2 values and converts the type to pd.Categorical

    Parameters
    ----------
    data: pd.DataFrame or pd.Series
        Data to be processed
    skip: str, list or None (default is None)
        List of variables that should *not* be made binary
    only: str, list or None (default is None)
        List of variables that are the *only* ones to be made binary

    Returns
    -------
    data: pd.DataFrame
        DataFrame with the same data but validated and converted to binary types

    Examples
    --------
    >>> import clarite
    >>> df = clarite.modify.make_binary(df)
    Set 32 of 32 variables as binary, each with 4,321 observations
    """
    # Which columns
    columns = _validate_skip_only(list(data), skip, only)

    # Check the number of unique values
    unique_values = data.nunique()
    num_non_binary = (unique_values[columns] > 2).sum()
    if num_non_binary > 0:
        raise ValueError(f"{num_non_binary} of {len(columns)} variables did not have 2 unique values and couldn't be processed as a binary type")
    # TODO: possibly add further validation to make sure values are 1 and 0

    # Convert dtype
    data = data.astype({c: 'category' for c in columns})
    print(f"Set {len(columns):,} of {len(data.columns)} variables as binary, each with {len(data):,} observations")

    return data


def make_categorical(data: pd.DataFrame, skip: Optional[Union[str, List[str]]] = None, only: Optional[Union[str, List[str]]] = None):
    """
    Validate and type a dataframe of categorical variables

    Converts the type to pd.Categorical

    Parameters
    ----------
    data: pd.DataFrame or pd.Series
        Data to be processed
    skip: str, list or None (default is None)
        List of variables that should *not* be made categorical
    only: str, list or None (default is None)
        List of variables that are the *only* ones to be made categorical

    Returns
    -------
    data: pd.DataFrame
        DataFrame with the same data but validated and converted to categorical types

    Examples
    --------
    >>> import clarite
    >>> df = clarite.modify.make_categorical(df)
    Set 12 of 12 variables as categorical, each with 4,321 observations
    """
    # Which columns
    columns = _validate_skip_only(list(data), skip, only)

    # TODO: possibly add further validation

    # Convert dtype
    data = data.astype({c: 'category' for c in columns})
    print(f"Set {len(columns):,} of {len(data.columns)} variables as categorical, each with {len(data):,} observations")

    return data


def make_continuous(data: pd.DataFrame, skip: Optional[Union[str, List[str]]] = None, only: Optional[Union[str, List[str]]] = None):
    """
    Validate and type a dataframe of continuous variables

    Converts the type to numeric

    Parameters
    ----------
    data: pd.DataFrame or pd.Series
        Data to be processed
    skip: str, list or None (default is None)
        List of variables that should *not* be made continuous
    only: str, list or None (default is None)
        List of variables that are the *only* ones to be made continuous

    Returns
    -------
    data: pd.DataFrame
        DataFrame with the same data but validated and converted to numeric types

    Examples
    --------
    >>> import clarite
    >>> df = clarite.modify.make_continuous(df)
    Set 128 of 128 variables as continuous, each with 4,321 observations
    """
    # Which columns
    columns = _validate_skip_only(list(data), skip, only)

    # TODO: possibly add further validation

    # Convert dtype
    data = pd.DataFrame({c: data[c] if c not in columns else pd.to_numeric(data[c]) for c in list(data)})
    print(f"Set {len(columns):,} of {len(data.columns)} variables as continuous, each with {len(data):,} observations")

    return data
