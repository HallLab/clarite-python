from typing import Optional, List

import pandas as pd
from scipy import stats


def _validate_skip_only(columns, skip: Optional[List[str]] = None, only: Optional[List[str]] = None):
    """Validate use of the 'skip' and 'only' parameters, returning a valid list of columns to filter"""
    if skip is not None and only is not None:
        raise ValueError("It isn't possible to specify 'skip' and 'only' at the same time.")
    elif skip is not None and only is None:
        invalid_cols = set(skip) - set(columns)
        if len(invalid_cols) > 0:
            raise ValueError(f"Invalid columns passed to 'skip': {', '.join(invalid_cols)}")
        columns = [c for c in columns if c not in set(skip)]
    elif skip is None and only is not None:
        invalid_cols = set(only) - set(columns)
        if len(invalid_cols) > 0:
            raise ValueError(f"Invalid columns passed to 'only': {', '.join(invalid_cols)}")
        columns = [c for c in columns if c in set(only)]

    if len(columns) == 0:
        raise ValueError("No columns available for filtering")

    return columns


def make_bin(df: pd.DataFrame):
    """
    Validate and type a dataframe of binary variables

    Checks that each variable has at most 2 values and converts the type to pd.Categorical

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame to be processed

    Returns
    -------
    df: pd.DataFrame
        DataFrame with the same data but validated and converted to categorical types

    Examples
    --------
    >>> df = clarite.make_bin(df)
    Processed 32 binary variables with 4,321 observations
    """
    # TODO: add further validation
    df = df.astype('category')
    print(f"Processed {len(df.columns):,} binary variables with {len(df):,} observations")
    return df


def make_categorical(df: pd.DataFrame):
    """
    Validate and type a dataframe of categorical variables

    Converts the type to pd.Categorical

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame to be processed

    Returns
    -------
    df: pd.DataFrame
        DataFrame with the same data but validated and converted to categorical types

    Examples
    --------
    >>> df = clarite.make_categorical(df)
    Processed 12 categorical variables with 4,321 observations
    """
    # TODO: add further validation
    df = df.astype('category')
    print(f"Processed {len(df.columns):,} categorical variables with {len(df):,} observations")
    return df


def make_continuous(df: pd.DataFrame):
    """
    Validate and type a dataframe of continuous variables

    Converts the type to numeric

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame to be processed

    Returns
    -------
    df: pd.DataFrame
        DataFrame with the same data but validated and converted to numeric types

    Examples
    --------
    >>> df = clarite.make_continuous(df)
    Processed 128 continuous variables with 4,321 observations
    """
    # TODO: add further validation
    df = df.apply(pd.to_numeric)
    print(f"Processed {len(df.columns):,} continuous variables with {len(df):,} observations")
    return df

def merge_variables(dataframes: List[pd.DataFrame]):
    """
    Merge dataframes with different variables side-by-side
    
    Parameters
    ----------
    dataframes: list of pd.Dataframe
        Dataframes to be merged.  Only observations present in all rows will be kept.
    """
    df = dataframes[0]
    for other in dataframes[1:]:
        df = df.merge(other, left_index=True, right_index=True)
    return df
