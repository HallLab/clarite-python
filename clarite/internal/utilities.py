from functools import wraps
from typing import Optional, List, Union

import click
import pandas as pd


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
    # Start with all as unknown
    dtypes = pd.Series('unknown', index=data.columns)

    # Set binary and categorical
    data_catbin = data.loc[:, data.dtypes == 'category']
    if len(data_catbin.columns) > 0:
        # Binary
        bin_cols = data_catbin.apply(lambda col: len(col.cat.categories) == 2)
        bin_cols = bin_cols[bin_cols].index
        dtypes.loc[bin_cols] = 'binary'
        # Categorical
        cat_cols = data_catbin.apply(lambda col: len(col.cat.categories) > 2)
        cat_cols = cat_cols[cat_cols].index
        dtypes.loc[cat_cols] = 'categorical'

    # Set continuous
    cont_cols = data.dtypes.apply(lambda dt: pd.api.types.is_numeric_dtype(dt))
    cont_cols = cont_cols[cont_cols].index
    dtypes.loc[cont_cols] = 'continuous'

    # Warn if there are any unknown types
    data_unknown = dtypes == 'unknown'
    unknown_num = data_unknown.sum()
    if unknown_num > 0:
        click.echo(click.style(f"WARNING: {unknown_num:,} variables need to be categorized into a type manually", fg='yellow'))

    return dtypes


def _process_colfilter(data: pd.DataFrame,
                       skip: Optional[Union[str, List[str]]],
                       only: Optional[Union[str, List[str]]],
                       fail_filter: pd.Series,  # Series mapping variable to a boolean of whether they failed the filter
                       explanation: str,  # A string explaining what the filter did (including any parameter values)
                       kinds: List[str]):  # Which variable types to apply the filter to
    """
    Log filter results, apply them to the data, and return the result.
    Saves a lot of repetitive code in column filtering functions.
    """
    columns = _validate_skip_only(data, skip, only)
    dtypes = _get_dtypes(data)

    kept = pd.Series(True, index=columns.index)

    for kind in kinds:
        is_kind = (dtypes == kind)
        is_tested_kind = is_kind & columns
        click.echo(f"Testing {is_tested_kind.sum():,} of {is_kind.sum():,} {kind} variables")
        removed_kind = is_tested_kind & fail_filter
        if is_tested_kind.sum() > 0:
            click.echo(f"\tRemoved {removed_kind.sum():,} ({removed_kind.sum()/is_tested_kind.sum():.2%}) "
                       f"tested {kind} variables {explanation}")
        kept = kept & ~removed_kind

    return kept
