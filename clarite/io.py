from pathlib import Path
from typing import Union
import pandas as pd


def load_data(filepath: Union[str, Path], index_col: str, **kwargs):
    """Load data from a file

    Wraps the Pandas 'read_csv' function but requires an index_col,
    and reports the number of variables and observations that were loaded.

    Parameters
    ----------
    filepath: str or path object
        File with data to be used in CLARITE
    index_col: int or string
        Column to use as the row labels of the DataFrame.
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
    df = pd.read_csv(filepath, index_col=index_col, **kwargs)
    print(f"Loaded {len(df):,} observations of {len(df.columns):,} variables")
    return df
