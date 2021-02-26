"""
Modify
======

Functions used to filter and/or change some data, always taking in one set of data and returning one set of data.

  .. autosummary::
     :toctree: modules/modify

     categorize
     colfilter
     colfilter_percent_zero
     colfilter_min_n
     colfilter_min_cat_n
     make_binary
     make_categorical
     make_continuous
     merge_observations
     merge_variables
     move_variables
     recode_values
     remove_outliers
     rowfilter_incomplete_obs
     transform

"""

from typing import Optional, List, Union

import click
import numpy as np
import pandas as pd

from ..internal.utilities import (
    _validate_skip_only,
    _get_dtypes,
    _process_colfilter,
    print_wrap,
    _remove_empty_categories,
)


@print_wrap
def categorize(
    data: pd.DataFrame, cat_min: int = 3, cat_max: int = 6, cont_min: int = 15
):
    """
    Classify variables into constant, binary, categorical, continuous, and 'unknown'.  Drop variables that only have NaN values.

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

    Examples
    --------
    >>> import clarite
    >>> clarite.modify.categorize(nhanes)
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
    assert type(data) == pd.DataFrame

    # Count the number of variables to start with
    total_vars = len(data.columns)
    # Get the number of unique non-na values per variable
    unique_count = data.nunique(dropna=True)

    # No unique non-NA values - Drop these variables
    empty_vars = unique_count == 0
    if empty_vars.sum() > 0:
        columns = list(empty_vars[empty_vars].index)
        data = data.drop(columns=columns)

    # One unique non-NA value - Convert non-NA values to category (for constant)
    keep_constant = unique_count == 1
    if keep_constant.sum() > 0:
        columns = list(keep_constant[keep_constant].index)
        data = data.astype({c: "category" for c in columns})

    # Two unique non-NA values - Convert non-NA values to category (for binary)
    keep_bin = unique_count == 2
    if keep_bin.sum() > 0:
        columns = list(keep_bin[keep_bin].index)
        data = data.astype({c: "category" for c in columns})

    # Categorical - Convert non-NA values to category type
    keep_cat = (unique_count >= cat_min) & (unique_count <= cat_max)
    if keep_cat.sum() > 0:
        columns = list(keep_cat[keep_cat].index)
        data = data.astype(
            {c: "category" for c in columns}
        )  # NaNs are handled correctly, no need skip

    # Continuous - Convert non-NA values to numeric type (even though they probably already are)
    keep_cont = unique_count >= cont_min
    check_cont = pd.Series(False, index=keep_cont.index)
    if keep_cont.sum() > 0:
        for col in keep_cont[keep_cont].index:
            try:
                data[col] = pd.to_numeric(data[col])
            except ValueError:
                # Couldn't convert to a number- possibly a categorical variable with string names?
                keep_cont[col] = False
                check_cont[col] = True
                data[col] = data.loc[:, col].astype(str)

    # Other - Convert non-NA values to string type
    check_other = (
        ~empty_vars & ~keep_constant & ~keep_bin & ~keep_cat & ~check_cont & ~keep_cont
    )
    if check_other.sum() > 0:
        columns = list(check_other[check_other].index)
        for c in columns:
            data.loc[~data[c].isna(), c] = data.loc[~data[c].isna(), c].astype(str)

    # Log categorized results
    click.echo(
        f"{keep_constant.sum():,} of {total_vars:,} variables ({keep_constant.sum() / total_vars:.2%}) "
        f"are classified as constant (1 unique value)."
    )
    click.echo(
        f"{keep_bin.sum():,} of {total_vars:,} variables ({keep_bin.sum()/total_vars:.2%}) "
        f"are classified as binary (2 unique values)."
    )
    click.echo(
        f"{keep_cat.sum():,} of {total_vars:,} variables ({keep_cat.sum()/total_vars:.2%}) "
        f"are classified as categorical ({cat_min} to {cat_max} unique values)."
    )
    click.echo(
        f"{keep_cont.sum():,} of {total_vars:,} variables ({keep_cont.sum()/total_vars:.2%}) "
        f"are classified as continuous (>= {cont_min} unique values)."
    )

    # Log dropped variables
    dropped = empty_vars.sum()
    click.echo(
        f"{dropped:,} of {total_vars:,} variables ({dropped/total_vars:.2%}) were dropped."
    )
    if dropped > 0:
        click.echo(f"\t{empty_vars.sum():,} variables had zero unique values (all NA).")

    # Log non-categorized results
    num_not_categorized = check_other.sum() + check_cont.sum()
    click.echo(
        f"{num_not_categorized:,} of {total_vars:,} variables ({num_not_categorized/total_vars:.2%})"
        f" were not categorized and need to be set manually."
    )
    if num_not_categorized > 0:
        click.echo(
            f"\t{check_other.sum():,} variables had between {cat_max} and {cont_min} unique values"
        )
        click.echo(
            f"\t{check_cont.sum():,} variables had >= {cont_min} values but couldn't be converted to continuous (numeric) values"
        )

    return data


@print_wrap
def colfilter(
    data,
    skip: Optional[Union[str, List[str]]] = None,
    only: Optional[Union[str, List[str]]] = None,
):
    """
    Remove some variables (skip) or keep only certain variables (only)

    Parameters
    ----------
    data: pd.DataFrame
        The DataFrame to be processed and returned
    skip: str, list or None (default is None)
        List of variables to remove
    only: str, list or None (default is None)
        List of variables to keep

    Returns
    -------
    data: pd.DataFrame
        The filtered DataFrame

    Examples
    --------
    >>> import clarite
    >>> female_logBMI = clarite.modify.colfilter(nhanes, only=['BMXBMI', 'female'])
    ================================================================================
    Running colfilter
    --------------------------------------------------------------------------------
    Keeping 2 of 945 variables:
            0 of 0 binary variables
            0 of 0 categorical variables
            2 of 945 continuous variables
            0 of 0 unknown variables
    ================================================================================
    """
    boolean_keep = _validate_skip_only(data, skip, only)
    dtypes = _get_dtypes(data)
    click.echo(f"Keeping {boolean_keep.sum():,} of {len(data.columns):,} variables:")

    for kind in ["binary", "categorical", "continuous", "unknown"]:
        is_kind = dtypes == kind
        is_kept = is_kind & boolean_keep
        click.echo(f"\t{is_kept.sum():,} of {is_kind.sum():,} {kind} variables")

    return data.loc[:, boolean_keep]


@print_wrap
def colfilter_min_cat_n(
    data,
    n: int = 200,
    skip: Optional[Union[str, List[str]]] = None,
    only: Optional[Union[str, List[str]]] = None,
):
    """
    Remove binary and categorical variables which have less than <n> occurences of each unique value

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
    >>> nhanes_filtered = clarite.modify.colfilter_min_cat_n(nhanes)
    ================================================================================
    Running colfilter_min_cat_n
    --------------------------------------------------------------------------------
    WARNING: 36 variables need to be categorized into a type manually
    Testing 362 of 362 binary variables
            Removed 248 (68.51%) tested binary variables which had a category with less than 200 values
    Testing 47 of 47 categorical variables
            Removed 36 (76.60%) tested categorical variables which had a category with less than 200 values
    """
    assert type(data) == pd.DataFrame
    min_category_counts = data.apply(lambda col: col.value_counts().min())
    fail_filter = min_category_counts < n

    kept = _process_colfilter(
        data,
        skip,
        only,
        fail_filter=fail_filter,
        explanation=f"which had a category with less than {n} values.",
        kinds=["binary", "categorical"],
    )
    # Return
    return data.loc[:, kept]


@print_wrap
def colfilter_min_n(
    data: pd.DataFrame,
    n: int = 200,
    skip: Optional[Union[str, List[str]]] = None,
    only: Optional[Union[str, List[str]]] = None,
):
    """
    Remove variables which have less than <n> non-NA values

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
    >>> nhanes_filtered = clarite.modify.colfilter_min_n(nhanes)
    ================================================================================
    Running colfilter_min_n
    --------------------------------------------------------------------------------
    WARNING: 36 variables need to be categorized into a type manually
    Testing 362 of 362 binary variables
            Removed 12 (3.31%) tested binary variables which had less than 200 non-null values
    Testing 47 of 47 categorical variables
            Removed 8 (17.02%) tested categorical variables which had less than 200 non-null values
    Testing 483 of 483 continuous variables
            Removed 8 (1.66%) tested continuous variables which had less than 200 non-null values
    """
    assert type(data) == pd.DataFrame
    counts = (
        data.count()
    )  # by default axis=0 (rows) so counts number of non-NA rows in each column
    fail_filter = counts < n

    kept = _process_colfilter(
        data,
        skip,
        only,
        fail_filter=fail_filter,
        explanation=f"which had less than {n} non-null values.",
        kinds=["binary", "categorical", "continuous"],
    )

    # Return
    return data.loc[:, kept]


@print_wrap
def colfilter_percent_zero(
    data: pd.DataFrame,
    filter_percent: float = 90.0,
    skip: Optional[Union[str, List[str]]] = None,
    only: Optional[Union[str, List[str]]] = None,
):
    """
    Remove continuous variables which have <proportion> or more values of zero (excluding NA)

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
    >>> nhanes_filtered = clarite.modify.colfilter_percent_zero(nhanes_filtered)
    ================================================================================
    Running colfilter_percent_zero
    --------------------------------------------------------------------------------
    WARNING: 36 variables need to be categorized into a type manually
    Testing 483 of 483 continuous variables
            Removed 30 (6.21%) tested continuous variables which were equal to zero in at least 90.00% of non-NA observations.
    """
    assert type(data) == pd.DataFrame
    percent_value = 100 * data.apply(lambda col: (col == 0).sum() / col.count())
    fail_filter = percent_value >= filter_percent

    kept = _process_colfilter(
        data,
        skip,
        only,
        fail_filter=fail_filter,
        explanation=f"which were equal to zero in at least {filter_percent:.2f}% of non-NA observations.",
        kinds=["continuous"],
    )
    # Return
    return data.loc[:, kept]


@print_wrap
def make_binary(
    data: pd.DataFrame,
    skip: Optional[Union[str, List[str]]] = None,
    only: Optional[Union[str, List[str]]] = None,
):
    """
    Set variable types as Binary

    Checks that each variable has at most 2 values and converts the type to pd.Categorical.

    Note: When these variables are used in regression, they are ordered by value.
    For example, Sex (Male=1, Female=2) will encode "Male" as 0 and "Female" as 1 during the EWAS regression step.

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
    >>> nhanes = clarite.modify.make_binary(nhanes, only=['female', 'black', 'mexican', 'other_hispanic'])
    ================================================================================
    Running make_binary
    --------------------------------------------------------------------------------
    Set 4 of 970 variable(s) as binary, each with 22,624 observations
    """
    # Which columns
    columns = _validate_skip_only(data, skip, only)

    # Check the number of unique values, raising an error if any columns can't be converted
    unique_values = data.nunique()
    non_binary = (unique_values != 2) & columns
    num_non_binary = non_binary.sum()
    if num_non_binary > 0:
        raise ValueError(
            f"{num_non_binary} variable(s) did not have 2 unique values and couldn't be processed as a binary type: "
            f"{', '.join(non_binary[non_binary].index)}"
        )

    # Convert dtype
    columns = columns[columns].index
    data = data.astype({c: "category" for c in columns})
    click.echo(
        f"Set {len(columns):,} of {len(data.columns):,} variable(s) as binary, each with {len(data):,} observations"
    )

    return data


@print_wrap
def make_categorical(
    data: pd.DataFrame,
    skip: Optional[Union[str, List[str]]] = None,
    only: Optional[Union[str, List[str]]] = None,
):
    """
    Set variable types as Categorical

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
    ================================================================================
    Running make_categorical
    --------------------------------------------------------------------------------
    Set 12 of 12 variable(s) as categorical, each with 4,321 observations
    """
    # Which columns
    columns = _validate_skip_only(data, skip, only)

    # Convert dtype
    columns = columns[columns].index
    data = data.astype({c: "category" for c in columns})
    click.echo(
        f"Set {len(columns):,} of {len(data.columns):,} variable(s) as categorical, each with {len(data):,} observations"
    )

    return data


@print_wrap
def make_continuous(
    data: pd.DataFrame,
    skip: Optional[Union[str, List[str]]] = None,
    only: Optional[Union[str, List[str]]] = None,
):
    """
    Set variable types as Numeric

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
    ================================================================================
    Running make_categorical
    --------------------------------------------------------------------------------
    Set 128 of 128 variable(s) as continuous, each with 4,321 observations
    """
    # Which columns
    columns = _validate_skip_only(data, skip, only)

    # Convert dtype, ignoring errors
    data.loc[:, columns] = data.loc[:, columns].apply(
        lambda col: pd.to_numeric(col, errors="ignore")
    )

    # Check if any variables could not be converted
    failed_conversion = data.loc[:, columns].dtypes.apply(
        lambda dt: not pd.api.types.is_numeric_dtype(dt)
    )
    if failed_conversion.sum() > 0:
        raise ValueError(
            f"{failed_conversion.sum()} variable(s) couldn't be processed as continuous (numeric) type(s): "
            f"{', '.join(failed_conversion[failed_conversion].index)}"
        )

    columns = columns[columns].index
    click.echo(
        f"Set {len(columns):,} of {len(data.columns):,} variable(s) as continuous, each with {len(data):,} observations"
    )

    return data


@print_wrap
def recode_values(
    data,
    replacement_dict,
    skip: Optional[Union[str, List[str]]] = None,
    only: Optional[Union[str, List[str]]] = None,
):
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
    >>> clarite.modify.recode_values(df, {7: np.nan, 9: np.nan}, only=['SMQ077', 'DBD100'])
    ================================================================================
    Running recode_values
    --------------------------------------------------------------------------------
    Replaced 17 values from 22,624 observations in 2 variables
    >>> clarite.modify.recode_values(df, {10: 12}, only=['SMQ077', 'DBD100'])
    ================================================================================
    Running recode_values
    --------------------------------------------------------------------------------
    No occurences of replaceable values were found, so nothing was replaced.
    """
    # Limit columns if needed
    if skip is not None or only is not None:
        columns = _validate_skip_only(data, skip, only)
        columns = columns[columns].index.get_level_values(
            0
        )  # variable names where columns = True
        replacement_dict = {c: replacement_dict for c in columns}

    # Replace
    result = data.replace(to_replace=replacement_dict, value=None, inplace=False)

    # Log
    diff = result.eq(data)
    diff[pd.isnull(result) & pd.isnull(data)] = True  # NAs are not equal by default
    diff = ~diff  # make True where a value was replaced
    cols_with_changes = (diff.sum() > 0).sum()
    cells_with_changes = diff.sum().sum()
    if cells_with_changes > 0:
        click.echo(
            f"Replaced {cells_with_changes:,} values from {len(data):,} observations in {cols_with_changes:,} variables"
        )
    else:
        click.echo(
            "No occurences of replaceable values were found, so nothing was replaced."
        )

    # Return
    return result


@print_wrap
def remove_outliers(
    data,
    method: str = "gaussian",
    cutoff=3,
    skip: Optional[Union[str, List[str]]] = None,
    only: Optional[Union[str, List[str]]] = None,
):
    """
    Remove outliers from continuous variables by replacing them with np.nan

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
    >>> nhanes_rm_outliers = clarite.modify.remove_outliers(nhanes, method='iqr', cutoff=1.5, only=['DR1TVB1', 'URXP07', 'SMQ077'])
    ================================================================================
    Running remove_outliers
    --------------------------------------------------------------------------------
    WARNING: 36 variables need to be categorized into a type manually
    Removing outliers from 2 continuous variables with values < 1st Quartile - (1.5 * IQR) or > 3rd quartile + (1.5 * IQR)
            Removed 0 low and 430 high IQR outliers from URXP07 (outside -153.55 to 341.25)
            Removed 0 low and 730 high IQR outliers from DR1TVB1 (outside -0.47 to 3.48)
    >>> nhanes_rm_outliers = clarite.modify.remove_outliers(nhanes, only=['DR1TVB1', 'URXP07'])
    ================================================================================
    Running remove_outliers
    --------------------------------------------------------------------------------
    WARNING: 36 variables need to be categorized into a type manually
    Removing outliers from 2 continuous variables with values more than 3 standard deviations from the mean
            Removed 0 low and 42 high gaussian outliers from URXP07 (outside -1,194.83 to 1,508.13)
            Removed 0 low and 301 high gaussian outliers from DR1TVB1 (outside -1.06 to 4.27)
    """
    # Copy to avoid replacing in-place
    data = data.copy(deep=True)

    # Which columns
    columns = _validate_skip_only(data, skip, only)
    is_continuous = _get_dtypes(data) == "continuous"
    columns = columns & is_continuous

    # Check cutoff and method, printing what is being done
    if cutoff <= 0:
        raise ValueError("'cutoff' must be >= 0")
    if method == "iqr":
        click.echo(
            f"Removing outliers from {len(data):,} observations of {columns.sum():,} continuous variables "
            f"with values < 1st Quartile - ({cutoff} * IQR) or > 3rd quartile + ({cutoff} * IQR)"
        )
    elif method == "gaussian":
        click.echo(
            f"Removing outliers from {len(data):,} observations of {columns.sum():,} continuous variables "
            f"with values more than {cutoff} standard deviations from the mean"
        )
    else:
        raise ValueError(
            f"'{method}' is not a supported method for outlier removal - only 'gaussian' and 'iqr'."
        )

    # Remove outliers
    # Note: This could be faster by performing calculations on the entire dataset at once, but in practice this should
    # be used on more of a limited basis, reviewing changes in each variable.
    for col_name, process_col in columns.iteritems():
        if not process_col:
            continue
        if method == "iqr":
            q1 = data[col_name].quantile(0.25)
            q3 = data[col_name].quantile(0.75)
            iqr = abs(q3 - q1)
            bottom = q1 - (iqr * cutoff)
            top = q3 + (iqr * cutoff)
        elif method == "gaussian":
            mean = data[col_name].mean()
            std = data[col_name].std()
            bottom = mean - (std * cutoff)
            top = mean + (std * cutoff)
        # Replace with NA
        outliers_bottom = data[col_name] < bottom
        outliers_top = data[col_name] > top
        data.loc[outliers_bottom, col_name] = np.nan
        data.loc[outliers_top, col_name] = np.nan
        # Log
        click.echo(
            f"\tRemoved {outliers_bottom.sum()} low and {outliers_top.sum()} high outliers "
            f"from {col_name} (outside {bottom:,.2f} to {top:,.2f})"
        )

    return data


@print_wrap
def rowfilter_incomplete_obs(
    data,
    skip: Optional[Union[str, List[str]]] = None,
    only: Optional[Union[str, List[str]]] = None,
):
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
    >>> nhanes_filtered = clarite.modify.rowfilter_incomplete_obs(nhanes, only=[outcome] + covariates)
    ================================================================================
    Running rowfilter_incomplete_obs
    --------------------------------------------------------------------------------
    Removed 3,687 of 22,624 observations (16.30%) due to NA values in any of 8 variables
    """
    columns = _validate_skip_only(data, skip, only)

    keep_IDs = (
        data.loc[:, columns].isnull().sum(axis=1) == 0
    )  # Number of NA in each row is 0
    n_removed = len(data) - sum(keep_IDs)

    click.echo(
        f"Removed {n_removed:,} of {len(data):,} observations ({n_removed/len(data):.2%}) "
        f"due to NA values in any of {columns.sum()} variables"
    )
    return data[keep_IDs]


@print_wrap
def merge_observations(top: pd.DataFrame, bottom: pd.DataFrame):
    """
    Merge two datasets, keeping only the columns present in both.  Raise an error if a datatype conflict occurs.

    Parameters
    ----------
    top: pd.DataFrame
        "top" DataFrame
    bottom: pd.DataFrame
        "bottom" DataFrame

    Returns
    -------
    result: pd.DataFrame
    """
    # Throw an error if any observation ID is repeated
    overlapped_observations = set(top.index) & set(bottom.index)
    if len(overlapped_observations) > 0:
        raise ValueError(
            f"Can't merge: {len(overlapped_observations):,} observation IDs occur in both datasets"
        )

    # Merge data, keeping only the columns in common
    combined = pd.concat([top, bottom], join="inner", sort=False)

    # If a categorical is only in one dataframe, or the categorical has different levels, it is coerced to an object and must be changed back
    # Exclude cases where either variable was an object originally
    combined = combined.astype(
        {
            col: "category"
            if (dt == "object")
            & (top.dtypes[col] != "object")
            & (bottom.dtypes[col] != "object")
            else dt
            for col, dt in combined.dtypes.iteritems()
        }
    )

    # Check datatypes for changes
    top_dtypes = _get_dtypes(top[combined.columns])
    bottom_dtypes = _get_dtypes(bottom[combined.columns])
    combined_dtypes = _get_dtypes(combined)

    diff_dtypes = top_dtypes != bottom_dtypes
    diff_dtype_vars = list(diff_dtypes[diff_dtypes].index)
    if diff_dtypes.sum() > 0:
        raise ValueError(
            f"{diff_dtypes.sum():,} variables have mismatched datatypes: \n{' '.join(diff_dtype_vars)}"
        )

    diff_cats = (top_dtypes != combined_dtypes) | (bottom_dtypes != combined_dtypes)
    diff_cats_vars = set(diff_cats[diff_cats].index) - set(diff_dtype_vars)
    if len(diff_cats_vars) > 0:
        raise ValueError(
            f"{len(diff_cats_vars):,} variables have categories present in only one dataset: \n{' '.join(diff_cats_vars)}"
        )

    return combined


@print_wrap
def merge_variables(
    left: Union[pd.DataFrame, pd.Series],
    right: Union[pd.DataFrame, pd.Series],
    how: str = "outer",
):
    """
    Merge a list of dataframes with different variables side-by-side.  Keep all observations ('outer' merge) by default.

    Parameters
    ----------
    left: pd.Dataframe or pd.Series
        "left" DataFrame or Series
    right: pd.DataFrame or pd.Series
        "right" DataFrame or Series which uses the same index
    how: merge method, one of {'left', 'right', 'inner', 'outer'}
        Keep only rows present in the left data, the right data, both datasets, or either dataset.

    Examples
    --------
    >>> import clarite
    >>> df = clarite.modify.merge_variables(df_bin, df_cat, how='outer')
    """
    # Convert to DataFrame if a series was passed
    if type(left) == pd.Series:
        left = pd.DataFrame(left)
    if type(right) == pd.Series:
        right = pd.DataFrame(right)

    click.echo(
        f"{how} Merge:\n"
        f"\tleft = {len(left):,} observations of {len(left.columns):,} variables\n"
        f"\tright = {len(right):,} observations of {len(right.columns):,} variables"
    )
    result = left.merge(right, left_index=True, right_index=True, how=how)
    click.echo(
        f"Kept {len(result):,} observations of {len(result.columns):,} variables."
    )
    return result


@print_wrap
def move_variables(
    left: pd.DataFrame,
    right: Union[pd.DataFrame, pd.Series],
    skip: Optional[Union[str, List[str]]] = None,
    only: Optional[Union[str, List[str]]] = None,
):
    """
    Move one or more variables from one DataFrame to another

    Parameters
    ----------
    left: pd.Dataframe
        DataFrame containing the variable(s) to be moved
    right: pd.DataFrame or pd.Series
        DataFrame or Series (which uses the same index) that the variable(s) will be moved to
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
    >>> df_cat, df_cont = clarity.modify.move_variables(df_cat, df_cont, only=["DRD350AQ", "DRD350DQ", "DRD350GQ"])
    Moved 3 variables.
    >>> discovery_check, discovery_cont = clarite.modify.move_variables(discovery_check, discovery_cont)
    Moved 39 variables.
    """
    # Which columns
    columns = _validate_skip_only(left, skip, only)

    # Add to new df
    right = merge_variables(right, left[columns])

    # Remove from original
    left = left.drop(columns, axis="columns")

    # Log
    if columns.sum() == 1:
        click.echo("Moved 1 variable.")
    else:
        click.echo(f"Moved {len(columns)} variables.")

    # Return
    return left, right


@print_wrap
def transform(
    data: pd.DataFrame,
    transform_method: str,
    skip: Optional[Union[str, List[str]]] = None,
    only: Optional[Union[str, List[str]]] = None,
):
    """
    Apply a transformation function to a variable

    Parameters
    ----------
    data: pd.DataFrame or pd.Series
        Data to be processed
    transform_method: str
        Name of the transformation (Python function or NumPy ufunc to apply)
    skip: str, list or None (default is None)
        List of variables that will *not* be transformed
    only: str, list or None (default is None)
        List of variables that are the *only* ones to be transformed

    Returns
    -------
    data: pd.DataFrame
        DataFrame with variables that have been transformed

    Examples
    --------
    >>> import clarite
    >>> df = clarite.modify.transform(df, 'log', only=['BMXBMI'])
    ================================================================================
    Running transform
    --------------------------------------------------------------------------------
    Transformed 'BMXBMI' using 'log'.
    """
    # Copy to avoid replacing in-place
    data = data.copy(deep=True)

    # Which columns
    columns = _validate_skip_only(data, skip, only)
    transform_variables = list(data.loc[:, columns])

    # Check for potential errors
    dtypes = _get_dtypes(data)
    for variable in transform_variables:
        dtype = dtypes.get(variable, None)
        if dtype is None:
            raise ValueError(f"The variable ('{variable}') was not found in the data")
        elif dtype != "continuous":
            raise ValueError(
                f"The variable ('{variable}') was {dtype}: "
                f"transformations may only be applied to continuous variables"
            )

    # Transform each variable
    for variable in transform_variables:
        try:
            data.loc[:, variable] = data.loc[:, variable].apply(transform_method)
        except Exception as e:
            raise ValueError(
                f"Couldn't apply a function named '{transform_method}' to '{variable}'.\n\t{e}"
            )

        click.echo(f"Transformed '{variable}' using '{transform_method}'")

    return data


@print_wrap
def drop_extra_categories(
    data: pd.DataFrame,
    skip: Optional[Union[str, List[str]]] = None,
    only: Optional[Union[str, List[str]]] = None,
):
    """
    Update variable types to remove categories that don't occur in the data

    Parameters
    ----------
    data: pd.DataFrame or pd.Series
        Data to be processed
    skip: str, list or None (default is None)
        List of variables that will *not* be checked
    only: str, list or None (default is None)
        List of variables that are the *only* ones to be checked

    Returns
    -------
    data: pd.DataFrame
        DataFrame with categorical types updated as needed

    Examples
    --------
    >>> import clarite
    >>> df = clarite.modify.drop_extra_categories(df, only=['SDDSRVYR'])
    ================================================================================
    Running drop_extra_categories
    --------------------------------------------------------------------------------
    SDDSRVYR had categories with no occurrences: 3, 4
    """
    # Copy to avoid replacing in-place
    data = data.copy(deep=True)

    # Drop categories
    removed_cats = _remove_empty_categories(data, skip, only)

    # Log results
    if len(removed_cats) == 1:
        var = list(removed_cats.keys())[0]
        cats = removed_cats[var]
        message = f"\t{str(var)} had categories with no occurrences: {', '.join([str(c) for c in cats])}"
        click.echo(message)
    elif len(removed_cats) > 1:
        message = "\tMultiple categorical variables had categories with no occurrences:"
        for var, cats in removed_cats.items():
            message += f"\n\t{str(var)}: {', '.join([str(c) for c in cats])}"
        click.echo(message)

    return data
