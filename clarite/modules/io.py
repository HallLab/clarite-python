"""
IO
========

Input/Output of data in different formats

  **DataFrame Accessor**: ``None``

  **CLI Command**: ``io``

  .. autosummary::
     :toctree: modules/io

     load_data
     save
     load_dtypes
     save_dtypes
"""

from pathlib import Path
import json
from typing import Optional, Union

import pandas as pd

from ..internal.utilities import get_dtypes, set_dtypes


def load_data(filename: str, index_col: Union[str, int] = 0, sep: str = '\t',
              dtypes_filename: Optional[str] = None, **kwargs):
    """Load data from a file

    Wraps the Pandas 'read_csv' function but requires an index_col,
    and reports the number of variables and observations that were loaded.

    Parameters
    ----------
    filename: str or Path
        File with data to be used in CLARITE
    index_col: int or string (default 0)
        Column to use as the row labels of the DataFrame.
    sep: str (default is "\t" for tab-separated)
        column separator (delimiter)
    dtypes_filename: optional string
        Name of the dtypes file for this data.  If not provided, check for filename + ".dtypes".  If that isn't found, don't change types from the default.
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
    # Load data
    data = pd.read_csv(filename, index_col=index_col, sep=sep, **kwargs)
    print(f"Loaded {len(data):,} observations of {len(data.columns):,} variables")

    # Load dtypes
    if dtypes_filename is None:
        dtypes_filename = str(filename) + ".dtypes"
    dtypes_filename = Path(dtypes_filename)
    if dtypes_filename.exists():
        # Try to load and set dtypes
        dtypes = load_dtypes(dtypes_filename)
        set_dtypes(data, dtypes)
    else:
        print("A dtypes file was not found, keeping default datatypes")

    return data


def load_dtypes(filename: str):
    """
    Load  a dtypes file

    Parameters
    ----------
    filename: str
        Name of the file to be loaded

    Returns
    -------
    Series
        A Series of datatypes

    Examples
    --------
    >>> clarite.io.load_dtypes('data/test_data.dtypes')
    """
    fpath = Path(filename)
    if not fpath.exists():
        raise ValueError(f"Could not read '{filename}'")
    else:
        with fpath.open('r') as f:
            try:
                dtypes = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"'{filename}' was not a valid dtypes file: {e}")
    return dtypes


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
    dtypes = get_dtypes(data)
    with open(filename, 'w') as f:
        json.dump(dtypes, f)


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
    save_dtypes(data, filename_dtypes)
