"""
Load
========

Load data from different formats or sources

     .. autofunction:: from_tsv
     .. autofunction:: from_csv
"""

from typing import Optional, Union

import click
import pandas as pd


def from_tsv(filename: str, index_col: Optional[Union[str, int]] = 0, **kwargs):
    """
    Load data from a tab-separated file into a DataFrame

    Parameters
    ----------
    filename: str or Path
        File with data to be used in CLARITE
    index_col: int or string (default 0)
        Column to use as the row labels of the DataFrame.
    **kwargs:
        Other keyword arguments to pass to pd.read_csv

    Returns
    -------
    DataFrame
        The index column will be used when merging

    Examples
    --------
    Load a tab-delimited file with an "ID" column

    >>> import clarite
    >>> df = clarite.load.from_tsv('nhanes.txt', index_col="SEQN")
    Loaded 22,624 observations of 970 variables
    """
    # Load data
    data = pd.read_csv(filename, index_col=index_col, sep="\t", **kwargs)
    # read_csv always returns a dataframe, so data.columns is okay here
    click.echo(f"Loaded {len(data):,} observations of {len(data.columns):,} variables")

    # Rename index to ID
    data.index.name = "ID"

    return data


def from_csv(filename: str, index_col: Optional[Union[str, int]] = 0, **kwargs):
    """
    Load data from a comma-separated file into a DataFrame

    Parameters
    ----------
    filename: str or Path
        File with data to be used in CLARITE
    index_col: int or string (default 0)
        Column to use as the row labels of the DataFrame.
    **kwargs:
        Other keyword arguments to pass to pd.read_csv

    Returns
    -------
    DataFrame
        The index column will be used when merging

    Examples
    --------
    Load a tab-delimited file with an "ID" column

    >>> import clarite
    >>> df = clarite.load.from_csv('nhanes.csv', index_col="SEQN")
    Loaded 22,624 observations of 970 variables
    """
    # Load data
    data = pd.read_csv(filename, index_col=index_col, **kwargs)
    click.echo(f"Loaded {len(data):,} observations of {len(data.columns):,} variables")

    # Rename index to ID
    data.index.name = "ID"

    return data
