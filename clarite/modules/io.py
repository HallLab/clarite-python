"""
IO
========

Input/Output of data in different formats

  **DataFrame Accessor**: ``None``

  **CLI Command**: ``io``

  .. autosummary::
     :toctree: modules/io

     load_data
"""

from pathlib import Path
from typing import Union
import pandas as pd


def load_data(filepath: Union[str, Path], index_col: Union[str, int] = 0, sep: str = '\t', **kwargs):
    """Load data from a file

    Wraps the Pandas 'read_csv' function but requires an index_col,
    and reports the number of variables and observations that were loaded.

    Parameters
    ----------
    filepath: str or path object
        File with data to be used in CLARITE
    index_col: int or string (default 0)
        Column to use as the row labels of the DataFrame.
    sep: str (default is "\t" for tab-separated)
        column separator (delimiter)
    **kwargs:
        Other keword arguments to pass to pd.read_csv

    Returns
    -------
    DataFrame
        The index column will be used when merging

    Examples
    --------
    Load a tab-delimited file with an "ID" column

    >>> df = io.load_data('nhanes.txt', index_col="ID", sep="\t")
    Loaded 22,624 observations of 970 variables
    """
    df = pd.read_csv(filepath, index_col=index_col, sep=sep, **kwargs)
    print(f"Loaded {len(df):,} observations of {len(df.columns):,} variables")
    return df


def save_dtypes(data: pd.DataFrame, filename: str):
    """
    Save a datatype file (.dtype) for the given data

    Parameters
    ----------
    data: pd.DataFrame
        Data to be saved
    filename: str
        Name of data file to be used in CLARITE - the 'dtypes' extension is added automatically if needed.

    Returns
    -------
    None

    Examples
    --------
    >>> clarite.io.save_dtypes(df, 'data/test_data')
    """
    # Check filename
    if not filename.endswith(".dtypes"):
        filename += ".dtypes"

    # Save
    data.dtypes.to_csv(filename, sep="\t", header=False)


def save(data: pd.DataFrame, filename: str):
    """
    Save a data to a file along with a dtype file

    Parameters
    ----------
    data: pd.DataFrame
        Data to be saved
    filename: str
        File with data to be used in CLARITE

    Returns
    -------
    None

    Examples
    --------
    >>> clarite.io.save(df, 'data/test_data')
    """
    # Check filename
    if not filename.endswith(".txt"):
        filename += ".txt"
    filename_dtypes = filename + ".dtypes"

    data.to_csv(filename, sep="\t", header=True)
    data.dtypes.to_csv(filename_dtypes, sep="\t", header=False)
