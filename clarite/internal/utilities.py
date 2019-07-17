from typing import Optional, List, Union

import pandas as pd
from pandas.api.types import is_numeric_dtype


def _validate_skip_only(columns, skip: Optional[Union[str, List[str]]] = None, only: Optional[Union[str, List[str]]] = None):
    """Validate use of the 'skip' and 'only' parameters, returning a valid list of columns to filter"""
    # Convert string to a list
    if type(skip) == str:
        skip = [skip]
    if type(only) == str:
        only = [only]

    if skip is not None and only is not None:
        raise ValueError(f"It isn't possible to specify 'skip' ({skip}) and 'only' ({only}) at the same time.")
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


def _split_data_by_type(data: pd.DataFrame):
    """Split a DataFrame into bin, cat, cont, and check"""
    data_catbin = data.loc[:, data.dtypes == 'category']
    data_bin = data_catbin.loc[:, data_catbin.apply(lambda col: len(col.cat.categories) == 2)]
    data_cat = data_catbin.loc[:, data_catbin.apply(lambda col: len(col.cat.categories) > 2)]
    data_cont = data.loc[:, data.apply(lambda col: is_numeric_dtype(col))]
    data_check = data.loc[:, data.dtypes == 'object']

    return (d if len(list(d)) > 0 else None for d in (data_bin, data_cat, data_cont, data_check))
