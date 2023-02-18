from functools import wraps
from importlib.util import find_spec
from typing import List, Optional, Union

import click
import pandas as pd
from pandas_genomics import GenotypeDtype

# GITHUB ISSUE #120: SettingWithCopyWarning on Regression runs
pd.set_option("mode.chained_assignment", None)


def print_wrap(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        click.echo("=" * 80)
        click.echo(f"Running {func.__name__}")
        click.echo("-" * 80)
        result = func(*args, **kwargs)
        click.echo("=" * 80)
        return result

    return wrapped


def requires(package_name):
    """Decorator factory to ensure optional packages are imported before running"""
    # Define and return an appropriate decorator
    def decorator(func):
        # Check if package is importable
        spec = find_spec(package_name)
        print(package_name, spec)
        if spec is None:

            @wraps(func)
            def wrapped(*args, **kwargs):
                raise ImportError(
                    f"Can't run '{func.__name__}' since '{package_name}' could not be imported"
                )

            return wrapped

        else:
            return func

    return decorator


def _validate_skip_only(
    data: pd.DataFrame,
    skip: Optional[Union[str, List[str]]] = None,
    only: Optional[Union[str, List[str]]] = None,
):
    """Validate use of the 'skip' and 'only' parameters, returning a boolean series for the columns where True = use the column"""
    # Ensure that 'data' is a DataFrame and not a Series
    if type(data) != pd.DataFrame:
        raise ValueError("The passed 'data' is not a Pandas DataFrame")

    # Convert string to a list
    if type(skip) == str:
        skip = [skip]
    if type(only) == str:
        only = [only]

    if skip is not None and only is not None:
        raise ValueError(
            "It isn't possible to specify 'skip' and 'only' at the same time."
        )
    elif skip is not None and only is None:
        invalid_cols = set(skip) - set(list(data))
        if len(invalid_cols) > 0:
            raise ValueError(
                f"Invalid columns passed to 'skip': {', '.join(invalid_cols)}"
            )
        columns = pd.Series(~data.columns.isin(skip), index=data.columns)
    elif skip is None and only is not None:
        invalid_cols = set(only) - set(list(data))
        if len(invalid_cols) > 0:
            raise ValueError(
                f"Invalid columns passed to 'only': {', '.join(invalid_cols)}"
            )
        columns = pd.Series(data.columns.isin(only), index=data.columns)
    else:
        columns = pd.Series(True, index=data.columns)

    if columns.sum() == 0:
        raise ValueError("No columns available for filtering")

    return columns


def _get_dtypes(data: pd.DataFrame):
    """Return a Series of CLARITE dtypes indexed by variable name"""
    # Ensure that 'data' is a DataFrame or Series (which is converted to a DataFrame)
    if isinstance(data, pd.Series):
        data = pd.DataFrame(data)
    if not isinstance(data, pd.DataFrame):
        raise ValueError("The passed 'data' is not a Pandas DataFrame or Series")

    # Start with all as unknown
    dtypes = pd.Series("unknown", index=data.columns)

    # Set genotype arrays
    gt_cols = data.apply(lambda col: GenotypeDtype.is_dtype(col))
    gt_cols = gt_cols[gt_cols].index
    dtypes.loc[gt_cols] = "genotypes"

    # Set binary and categorical
    data_catbin = data.loc[:, data.dtypes == "category"]
    if len(data_catbin.columns) > 0:
        # Constant
        constant_cols = data_catbin.apply(lambda col: len(col.cat.categories) == 1)
        constant_cols = constant_cols[constant_cols].index
        dtypes.loc[constant_cols] = "constant"
        # Binary
        bin_cols = data_catbin.apply(lambda col: len(col.cat.categories) == 2)
        bin_cols = bin_cols[bin_cols].index
        dtypes.loc[bin_cols] = "binary"
        # Categorical
        cat_cols = data_catbin.apply(lambda col: len(col.cat.categories) > 2)
        cat_cols = cat_cols[cat_cols].index
        dtypes.loc[cat_cols] = "categorical"

    # Set continuous
    cont_cols = data.dtypes.apply(lambda dt: pd.api.types.is_numeric_dtype(dt))
    cont_cols = cont_cols[cont_cols].index
    dtypes.loc[cont_cols] = "continuous"

    # Warn if there are any unknown types
    data_unknown = dtypes == "unknown"
    unknown_num = data_unknown.sum()
    if unknown_num > 0:
        click.echo(
            click.style(
                f"WARNING: {unknown_num:,} variables need to be categorized into a type manually",
                fg="yellow",
            )
        )

    return dtypes


def _get_dtype(data: pd.Series):
    """Return the CLARITE dtype of a pandas series"""
    # Set binary and categorical
    if GenotypeDtype.is_dtype(data):
        return "genotypes"
    elif data.dtype.name == "category":
        num_categories = len(data.cat.categories)
        if num_categories == 1:
            return "constant"
        elif num_categories == 2:
            return "binary"
        elif num_categories > 2:
            return "categorical"
    elif pd.api.types.is_numeric_dtype(data.dtype):
        return "continuous"
    else:
        return "unknown"


def _process_colfilter(
    data: pd.DataFrame,
    skip: Optional[Union[str, List[str]]],
    only: Optional[Union[str, List[str]]],
    fail_filter: pd.Series,  # Series mapping variable to a boolean of whether they failed the filter
    explanation: str,  # A string explaining what the filter did (including any parameter values)
    kinds: List[str],
):  # Which variable types to apply the filter to
    """
    Log filter results, apply them to the data, and return the result.
    Saves a lot of repetitive code in column filtering functions.
    """
    columns = _validate_skip_only(data, skip, only)
    dtypes = _get_dtypes(data)

    kept = pd.Series(True, index=columns.index)

    for kind in kinds:
        is_kind = dtypes == kind
        is_tested_kind = is_kind & columns
        click.echo(
            f"Testing {is_tested_kind.sum():,} of {is_kind.sum():,} {kind} variables"
        )
        removed_kind = is_tested_kind & fail_filter
        if is_tested_kind.sum() > 0:
            click.echo(
                f"\tRemoved {removed_kind.sum():,} ({removed_kind.sum()/is_tested_kind.sum():.2%}) "
                f"tested {kind} variables {explanation}"
            )
        kept = kept & ~removed_kind

    return kept


def _remove_empty_categories(
    data: Union[pd.DataFrame, pd.Series],
    skip: Optional[Union[str, List[str]]] = None,
    only: Optional[Union[str, List[str]]] = None,
):
    """
    Remove categories from categorical types if there are no occurrences of that type.
    Updates the data in-place and returns a dict of variables:removed categories
    """
    removed_cats = dict()
    if type(data) == pd.DataFrame:
        columns = _validate_skip_only(data, skip, only)
        dtypes = data.loc[:, columns].dtypes
        catvars = [v for v in dtypes[dtypes == "category"].index]
        for var in catvars:
            existing_cats = data[var].cat.categories
            if data[var].cat.ordered:
                print()
            # GITHUB ISSUE #120: SettingWithCopyWarning on Regression runs
            data[var] = data[var].cat.remove_unused_categories()
            # data.loc[:, var] = data[var].cat.remove_unused_categories()
            removed_categories = set(existing_cats) - set(data[var].cat.categories)
            if len(removed_categories) > 0:
                removed_cats[var] = removed_categories
        return removed_cats
    elif type(data) == pd.Series:
        assert skip is None
        assert only is None
        counts = data.value_counts()
        keep_cats = list(counts[counts > 0].index)
        if len(keep_cats) < len(counts):
            removed_cats[data.name] = set(counts.index) - set(keep_cats)
            data.cat.set_categories(
                new_categories=keep_cats, ordered=data.cat.ordered, inplace=True
            )
        return removed_cats
