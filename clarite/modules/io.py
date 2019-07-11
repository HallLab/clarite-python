"""
IO
========

Input/Output of data in different formats

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


def load_data(filename: str, index_col: Optional[Union[str, int]] = 0, sep: str = '\t',
              dtypes: Optional[Union[str, bool]] = None, **kwargs):
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
    dtypes: optional string or boolean
        Name of the dtypes file for this data.
         If not provided, check for the default dtypes file (filename + ".dtypes") and use it if available.
         If a boolean, require that the default dtypes file is available (True) or don't use it (False).
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

    # Rename index to ID
    data.index.name = "ID"

    # Get default dtypes filename if one wasn't provided
    if dtypes is None or dtypes is True or dtypes is False:
        dtypes_filename = str(filename) + ".dtypes"
    else:
        dtypes_filename = dtypes
    dtypes_file = Path(dtypes_filename)

    # Depending on the input parameter, handle dtypes
    if dtypes is False:
        # Don't load the file no matter what
        print("\tIgnoring any existing dtypes file, using default datatypes instead")
    elif dtypes is None:
        # Try to load the default filename, but do nothing if not found
        if dtypes_file.exists():
            dtypes = load_dtypes(dtypes_filename)
            set_dtypes(data, dtypes)
            print(f"\tLoaded dtypes file: {dtypes_filename}")
    elif dtypes is True or type(dtypes) == str:
        # Throw an error if the file isn't found (True, or a specific file)
        if dtypes_file.exists():
            dtypes = load_dtypes(dtypes_filename)
            set_dtypes(data, dtypes)
        else:
            raise ValueError(f"Couldn't load dtypes file: {dtypes_filename}")

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

    if len(data.columns) > 0:
        data.to_csv(filename, sep="\t", header=True)
        save_dtypes(data, filename_dtypes)
