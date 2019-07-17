from functools import wraps
from typing import Optional, List, Union

import click
import pandas as pd
from pandas.api.types import is_numeric_dtype


def print_wrap(func):
    @wraps(func)
    def wrapper_do_twice(*args, **kwargs):
        console_width, _ = click.get_terminal_size()
        click.echo("=" * console_width)
        click.echo(f"Running {func.__name__}")
        click.echo("-" * console_width)
        return func(*args, **kwargs)
    return wrapper_do_twice


def _validate_skip_only(data: pd.DataFrame, skip: Optional[Union[str, List[str]]] = None, only: Optional[Union[str, List[str]]] = None):
    """Validate use of the 'skip' and 'only' parameters, returning a boolean series for the columns where True = use the column"""
    # Convert string to a list
    if type(skip) == str:
        skip = [skip]
    if type(only) == str:
        only = [only]

    if skip is not None and only is not None:
        raise ValueError(f"It isn't possible to specify 'skip' ({skip}) and 'only' ({only}) at the same time.")
    elif skip is not None and only is None:
        invalid_cols = set(skip) - set(list(data))
        if len(invalid_cols) > 0:
            raise ValueError(f"Invalid columns passed to 'skip': {', '.join(invalid_cols)}")
        columns = pd.Series(~data.columns.isin(skip), index=data.columns)
    elif skip is None and only is not None:
        invalid_cols = set(only) - set(list(data))
        if len(invalid_cols) > 0:
            raise ValueError(f"Invalid columns passed to 'only': {', '.join(invalid_cols)}")
        columns = pd.Series(data.columns.isin(only), index=data.columns)
    else:
        columns = pd.Series(True, index=data.columns)

    if columns.sum() == 0:
        raise ValueError("No columns available for filtering")

    return columns


def _get_dtypes(data: pd.DataFrame):
    """Return a Series of dtypes indexed by variable name"""
    dtypes = data.dtypes

    data_catbin = data.loc[:, data.dtypes == 'category']
    data_bin = data_catbin.loc[:, data_catbin.apply(lambda col: len(col.cat.categories) == 2)]
    data_cat = data_catbin.loc[:, data_catbin.apply(lambda col: len(col.cat.categories) > 2)]
    data_cont = data.loc[:, data.apply(lambda col: is_numeric_dtype(col))]
    data_unknown = data.loc[:, data.dtypes == 'object']

    dtypes.loc[data_bin.columns] = 'binary'
    dtypes.loc[data_cat.columns] = 'categorical'
    dtypes.loc[data_cont.columns] = 'continuous'
    dtypes.loc[data_unknown.columns] = 'unknown'

    unknown_num = len(list(data_unknown))
    if unknown_num > 0:
        click.echo(click.style(f"\tWARNING: {unknown_num:,} variables need to be categorized into a type manually", fg='yellow'))

    return dtypes


def _process_colfilter(data: pd.DataFrame,
                       skip: Optional[Union[str, List[str]]],
                       only: Optional[Union[str, List[str]]],
                       fail_filter: pd.Series,  # Series mapping variable to a boolean of whether they failed the filter
                       explanation: str,  # A string explaining what the filter did (including any parameter values)
                       kinds: List[str]):  # Which variable types to apply the filter to
    """
    Log filter results, apply them to the data, and return the result.
    Saves a lot of repetitive code.
    """
    columns = _validate_skip_only(data, skip, only)
    dtypes = _get_dtypes(data)

    kept = pd.Series(True, index=columns.index)

    for kind in kinds:
        is_kind = (dtypes == kind)
        is_tested_kind = is_kind & columns
        click.echo(f"\tTesting {is_tested_kind.sum():,} of {is_kind.sum():,} {kind} variables")
        removed_kind = is_tested_kind & fail_filter
        if is_tested_kind.sum() > 0:
            click.echo(f"\t\tRemoved {removed_kind.sum():,} ({removed_kind.sum()/is_tested_kind.sum():.2%}) "
                       f"tested {kind} variables {explanation}")
        kept = kept & ~removed_kind

    return kept
