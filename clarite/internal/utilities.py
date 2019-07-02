from typing import Optional, List

import numpy as np
import pandas as pd


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


def get_dtypes(data):
    """
    Convert dtypes of a DataFrame or Series to a dictionary format
    Examples:
      binary: {'female': {'type': 'category', 'categories': [0, 1], 'ordered': False}}
      categorical: {'CALCIUM_Unknown': {'type': 'category', 'categories': [0.0, 0.066666666, 0.933333333], 'ordered': False}}
      continuous: {'BMXBMI': {'type': 'float64'}}
    """
    dtypes = {variable_name: {'type': str(dtype)} if str(dtype) != 'category'
              else {'type': str(dtype), 'categories': list(dtype.categories.values.tolist()), 'ordered': dtype.ordered}
              for variable_name, dtype in data.dtypes.iteritems()}
    return dtypes


def set_dtypes(data, dtypes):
    """
    Set the dtypes of a dataframe according to a dtypes dictionary (in-place)
    """
    # Validate
    missing_types = set(list(data)) - set(dtypes.keys())
    extra_dtypes = set(dtypes.keys()) - set(list(data))
    if len(missing_types) > 0:
        raise ValueError(f"Dtypes file is missing some values: {', '.join(missing_types)}")
    if len(extra_dtypes) > 0:
        raise ValueError(f"Dtypes file has types for variables not found in the data: {', '.join(extra_dtypes)}")

    for col in list(data):
        typeinfo = dtypes[col]
        newtype = typeinfo['type']
        if typeinfo['type'] == 'category':
            newtype = pd.CategoricalDtype(categories=np.array(typeinfo['categories']), ordered=typeinfo['ordered'])
        data[col] = data[col].astype(newtype)
