"""
Describe
========

Functions that are used to gather information about some data

  .. autosummary::
     :toctree: modules/describe

     correlations
     freq_table
     get_types
     percent_na
     skewness
     summarize

"""

# Describe - functions that are used to gather information about some data

import click
import numpy as np
import pandas as pd
from scipy import stats

from ..internal.utilities import _get_dtypes


def correlations(data: pd.DataFrame, threshold: float = 0.75):
    """
    Return variables with pearson correlation above the threshold

    Parameters
    ----------
    data: pd.DataFrame
        The DataFrame to be described
    threshold: float, between 0 and 1
        Return a dataframe listing pairs of variables whose absolute value of correlation is above this threshold

    Returns
    -------
    result: pd.DataFrame
        DataFrame listing pairs of correlated variables and their correlation value

    Examples
    --------
    >>> import clarite
    >>> correlations = clarite.describe.correlations(df, threshold=0.9)
    >>> correlations.head()
                        var1      var2  correlation
    0  supplement_count  DSDCOUNT     1.000000
    1          DR1TM181  DR1TMFAT     0.997900
    2          DR1TP182  DR1TPFAT     0.996172
    3          DRD370FQ  DRD370UQ     0.987974
    4          DR1TS160  DR1TSFAT     0.984733
    """
    assert type(data) == pd.DataFrame
    # Get correlaton matrix
    correlation = data.corr()
    # Keep only the upper triangle to avoid listing both a-b and b-a correlations
    correlation = correlation.where(
        np.triu(np.ones(correlation.shape), k=1).astype(bool)
    )
    # Stack and rename into the desired format
    correlation = (
        correlation.stack()
        .rename("correlation")
        .rename_axis(["var1", "var2"])
        .reset_index()
    )
    # Remove those with correlation below threshold
    correlation = correlation.loc[
        correlation["correlation"].abs() >= threshold,
    ]
    # Sort by absolute value
    correlation = correlation.reindex(
        correlation["correlation"].abs().sort_values(ascending=False).index
    )
    # Return with a reset index
    return correlation.reset_index(drop=True)


def freq_table(data: pd.DataFrame):
    """
    Return the count of each unique value for all binary and categorical variables.  Other variables
    will return a single row with a value of '<Non-Categorical Values>' and the number of non-NA values.

    Parameters
    ----------
    data: pd.DataFrame
        The DataFrame to be described

    Returns
    -------
    result: pd.DataFrame
        DataFrame listing variable, value, and count for each categorical variable

    Examples
    --------
    >>> import clarite
    >>> clarite.describe.freq_table(df).head(n=10)
        variable value  count
    0                 SDDSRVYR                         2   4872
    1                 SDDSRVYR                         1   4191
    2                   female                         1   4724
    3                   female                         0   4339
    4  how_many_years_in_house                         5   2961
    5  how_many_years_in_house                         3   1713
    6  how_many_years_in_house                         2   1502
    7  how_many_years_in_house                         1   1451
    8  how_many_years_in_house                         4   1419
    9                  LBXPFDO  <Non-Categorical Values>   1032
    """
    assert type(data) == pd.DataFrame

    # Define a function to be applied to each categorical variable
    def formatted_value_counts(var_name: str, df: pd.DataFrame):
        if str(df[var_name].dtype) == "category":
            # Binary and categorical variables
            df = (
                df[var_name]
                .value_counts()
                .reset_index()
                .rename({"index": "value", var_name: "count"}, axis="columns")
            )
            df["variable"] = var_name
            return df[["variable", "value", "count"]]  # reorder columns
        else:
            # Continuous or "check" variables
            return pd.DataFrame.from_dict(
                {
                    "variable": [var_name],
                    "value": ["<Non-Categorical Values>"],
                    "count": [df[var_name].count()],
                }
            )

    return pd.concat(
        [formatted_value_counts(var_name, data) for var_name in list(data)]
    ).reset_index(drop=True)


def get_types(data: pd.DataFrame):
    """
    Return the type of each variable

    Parameters
    ----------
    data: pd.DataFrame
        The DataFrame to be described

    Returns
    -------
    result: pd.Series
        Series listing the CLARITE type for each variable

    Examples
    --------
    >>> import clarite
    >>> clarite.describe.get_types(df).head()
    RIDAGEYR          continuous
    female                binary
    black                 binary
    mexican               binary
    other_hispanic        binary
    dtype: object
    """
    return _get_dtypes(data)


def percent_na(data: pd.DataFrame):
    """
    Return the percent of observations that are NA for each variable

    Parameters
    ----------
    data: pd.DataFrame
        The DataFrame to be described

    Returns
    -------
    result: pd.DataFrame
        DataFrame listing percent NA for each variable

    Examples
    --------
    >>> import clarite
    >>> clarite.describe.percent_na(df)
       variable  percent_na
    0  SDDSRVYR     0.00000
    1    female     0.00000
    2    LBXHBC     4.99321
    3    LBXHBS     4.98730
    """
    assert type(data) == pd.DataFrame
    result = 100 * (1 - (data.count() / data.apply(len)))
    result = result.reset_index()
    result.columns = ["Variable", "percent_na"]
    return result


def skewness(data: pd.DataFrame, dropna: bool = False):
    """
    Return the skewness of each continuous variable

    Parameters
    ----------
    data: pd.DataFrame
        The DataFrame to be described
    dropna: bool
        If True, drop rows with NA values before calculating skew.  Otherwise the NA values propagate.

    Returns
    -------
    result: pd.DataFrame
        DataFrame listing three values for each continuous variable and NA for others: skew, zscore, and pvalue
        The test null hypothesis is that the skewness of the samples population is the same as the corresponding
         normal distribution.  The pvalue is the two-sided pvalue for the hypothesis test

    Examples
    --------
    >>> import clarite
    >>> clarite.describe.skewness(df)
         Variable         type      skew    zscore        pvalue
    0       pdias  categorical       NaN       NaN           NaN
    1   longindex  categorical       NaN       NaN           NaN
    2     durflow   continuous  2.754286  8.183515  2.756827e-16
    3      height   continuous  0.583514  2.735605  6.226567e-03
    4     begflow   continuous -0.316648 -1.549449  1.212738e-01
    """
    # Get continuous variables
    dtypes = _get_dtypes(data)
    continuous_idx = dtypes[dtypes == "continuous"].index

    # Format result df, starting with NA
    result = pd.DataFrame(
        data=None,
        index=dtypes.index,
        columns=["type", "skew", "zscore", "pvalue"],
        dtype=float,
    )
    result["type"] = dtypes

    # Calculate skew and statistical test
    if dropna:
        nan_policy = "omit"
    else:
        nan_policy = "propagate"
    result["skew"] = stats.skew(data[continuous_idx], nan_policy=nan_policy)
    (
        result.loc[continuous_idx, "zscore"],
        result.loc[continuous_idx, "pvalue"],
    ) = stats.skewtest(data[continuous_idx], nan_policy=nan_policy)

    # Format
    result.index.name = "Variable"
    result = result.reset_index()
    return result


def summarize(data: pd.DataFrame):
    """
    Print the number of each type of variable and the number of observations

    Parameters
    ----------
    data: pd.DataFrame
        The DataFrame to be described

    Returns
    -------
    result: None

    Examples
    --------
    >>> import clarite
    >>> clarite.describe.get_types(df).head()
    RIDAGEYR          continuous
    female                binary
    black                 binary
    mexican               binary
    other_hispanic        binary
    dtype: object
    """
    type_counts = _get_dtypes(data).value_counts()
    click.echo(
        f"{len(data):,} observations of {len(data.columns):,} variables\n"
        f"\t{type_counts.get('binary', 0):,} Binary Variables\n"
        f"\t{type_counts.get('categorical', 0):,} Categorical Variables\n"
        f"\t{type_counts.get('continuous', 0):,} Continuous Variables\n"
        f"\t{type_counts.get('unknown', 0):,} Unknown-Type Variables\n"
    )
