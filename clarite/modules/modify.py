# Modify - functions that are used to modify data and return it in the same form

from typing import Optional, List

import pandas as pd

from ..other.utilities import _validate_skip_only


def colfilter_percent_zero(data: pd.DataFrame, proportion: float = 0.9,
                           skip: Optional[List[str]] = None, only: Optional[List[str]] = None):
    """
    Remove columns which have <proportion> or more values of zero (excluding NA)

    Parameters
    ----------
    data: pd.DataFrame
        The DataFrame to be processed and returned
    proportion: float, default 0.9
        If the proportion of rows in the data with a value of zero is greater than or equal to this value, the variable is filtered out.
    skip: list or None, default None
        List of variables that the filter should *not* be applied to
    only: list or None, default None
        List of variables that the filter should *only* be applied to

    Returns
    -------
    data: pd.DataFrame
        The filtered DataFrame

    Examples
    --------
    import clarite
    >>> nhanes_discovery_cont = clarite.modify.colfilter_percent_zero(nhanes_discovery_cont)
    Removed 30 of 369 variables (8.13%) which were equal to zero in at least 90.00% of non-NA observations.
    """
    columns = _validate_skip_only(list(data), skip, only)
    num_before = len(data.columns)

    percent_value = data.apply(lambda col: sum(col == 0) / col.count())
    kept = (percent_value < proportion) | ~data.columns.isin(columns)
    num_removed = num_before - sum(kept)

    print(f"Removed {num_removed:,} of {num_before:,} variables ({num_removed/num_before:.2%}) "
          f"which were equal to zero in at least {proportion:.2%} of non-NA observations.")
    return data.loc[:, kept]


def colfilter_min_n(data: pd.DataFrame, n: int = 200,
                    skip: Optional[List[str]] = None, only: Optional[List[str]] = None):
    """
    Remove columns which have less than <n> unique values (excluding NA)

    Parameters
    ----------
    data: pd.DataFrame
        The DataFrame to be processed and returned
    n: int, default 200
        The minimum number of unique values required in order for a variable not to be filtered
    skip: list or None, default None
        List of variables that the filter should *not* be applied to
    only: list or None, default None
        List of variables that the filter should *only* be applied to

    Returns
    -------
    data: pd.DataFrame
        The filtered DataFrame

    Examples
    --------
    import clarite
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


def colfilter_min_cat_n(data, n: int = 200, skip: Optional[List[str]] = None, only: Optional[List[str]] = None):
    """
    Remove columns which have less than <n> occurences of each unique value

    Parameters
    ----------
    data: pd.DataFrame
        The DataFrame to be processed and returned
    n: int, default 200
        The minimum number of occurences of each unique value required in order for a variable not to be filtered
    skip: list or None, default None
        List of variables that the filter should *not* be applied to
    only: list or None, default None
        List of variables that the filter should *only* be applied to

    Returns
    -------
    data: pd.DataFrame
        The filtered DataFrame

    Examples
    --------
    import clarite
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


def rowfilter_incomplete_observations(data, skip, only):
    """
    Remove rows containing null values

    Parameters
    ----------
    data: pd.DataFrame
        The DataFrame to be processed and returned
    skip: list or None, default None
        List of columns that are not checked for null values
    only: list or None, default None
        List of columns that are the only ones to be checked for null values

    Returns
    -------
    data: pd.DataFrame
        The filtered DataFrame

    Examples
    --------
    >>> nhanes = clarite.modify.rowfilter_incomplete_observations(only=[phenotype] + covariates)
    Removed 3,687 of 22,624 rows (16.30%) due to NA values in the specified columns
    """
    columns = _validate_skip_only(list(data), skip, only)

    keep_IDs = data[columns].isnull().sum(axis=1) == 0  # Number of NA in each row is 0
    n_removed = len(data) - sum(keep_IDs)

    print(f"Removed {n_removed:,} of {len(data):,} rows ({n_removed/len(data):.2%}) due to NA values in the specified columns")
    return data[keep_IDs]


def recode_values(self, to_replace, value, inplace=False, skip: Optional[List[str]] = None, only: Optional[List[str]] = None):
        # TODO: Rewrite this
        """
        Convert one value to another.  By default, occurs in all columns but this may be modified with 'skip' or 'only'.
        A simpler, more verbose, less-powerful version of the Pandas 'df.replace' method.

        Parameters
        ----------
        to_replace: str, int, float, or dict
            The value to be replaced.  A dict may be used to make multiple replacements.
        value: str, int, float, or None.
            The value used to replace the original.  Must be None when 'to_replace' is a dict.
        inplace: boolean (default = False)
            If True, modify the dataframe in-place and return nothing.  If False, copy the dataframe and return the new copy.
        skip: list or None, default None
            List of variables that the replacement should *not* be applied to
        only: list or None, default None
            List of variables that the replacement should *only* be applied to

        Examples
        --------
        >>> df.clarite.recode_values(7, np.nan, only=['SMQ077', 'DBD100'], inplace=True)
        Replacing '7' with 'nan' in 2 columns.
            Replaced 1 of 18,937 rows of SMQ077
        >>> df.clarite.recode_values(7, np.nan, only=['SMQ077', 'DBD100'], inplace=True)
        Replacing '7' with 'nan' in 2 columns.
            No occurences of '7' were found, so nothing was replaced.
        """
        # Validate
        if type(to_replace) == dict and value is not None:
            raise ValueError(f"When 'to_replace' is a dictionary, 'value' must be None.")

        # In place or not
        if inplace:
            df = self._obj
        else:
            df = self._obj.copy(deep=True)

        if type(to_replace) == dict:
            # Recursively replace
            for k, v in to_replace.items():
                df.clarite.recode_values(k, v, skip=skip, only=only, inplace=True)
            return df
        else:
            unchanged = True
            columns = _validate_skip_only(list(df), skip, only)
            print(f"Replacing '{to_replace}' with '{value}' in {len(columns)} columns.")
            for c in columns:
                replaced = (df[c] == to_replace)
                df.loc[replaced, c] = value
                if sum(replaced) > 0:
                    unchanged = False
                    print(f"\tReplaced {sum(replaced):,} of {len(df):,} rows of {c}")
            if unchanged:
                print(f"\tNo occurences of '{to_replace}' were found, so nothing was replaced.")

        # Return the dataframe if not modified in place
        if not inplace:
            return df