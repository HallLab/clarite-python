from functools import wraps
from importlib.util import find_spec
from typing import Optional, List, Union

import click
import pandas as pd


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
        if spec is None:

            @wraps(func)
            def wrapped(*args, **kwargs):
                raise ImportError(f"Can't run '{func.__name__}' since '{package_name}' could not be imported")
            return wrapped

        else:
            return func

    return decorator


def _validate_skip_only(
        data: pd.DataFrame,
        skip: Optional[Union[str, List[str]]] = None,
        only: Optional[Union[str, List[str]]] = None):
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
        raise ValueError("It isn't possible to specify 'skip' and 'only' at the same time.")
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
    """Return a Series of CLARITE dtypes indexed by variable name"""
    # Ensure that 'data' is a DataFrame and not a Series
    if type(data) != pd.DataFrame:
        raise ValueError("The passed 'data' is not a Pandas DataFrame")

    # Start with all as unknown
    dtypes = pd.Series('unknown', index=data.columns)

    # Set binary and categorical
    data_catbin = data.loc[:, data.dtypes == 'category']
    if len(data_catbin.columns) > 0:
        # Constant
        constant_cols = data_catbin.apply(lambda col: len(col.cat.categories) == 1)
        constant_cols = constant_cols[constant_cols].index
        dtypes.loc[constant_cols] = 'constant'
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


def _get_dtype(data: pd.Series):
    """Return the CLARITE dtype of a pandas series"""
    # Set binary and categorical
    if data.dtype.name == 'category':
        num_categories = len(data.cat.categories)
        if num_categories == 1:
            return 'constant'
        elif num_categories == 2:
            return 'binary'
        elif num_categories > 2:
            return 'categorical'
    elif pd.api.types.is_numeric_dtype(data.dtype):
        return 'continuous'
    else:
        return 'unknown'


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


def _remove_empty_categories(data: pd.DataFrame,
                             skip: Optional[Union[str, List[str]]] = None,
                             only: Optional[Union[str, List[str]]] = None):
    """
    Remove categories from categorical types if there are no occurrences of that type.
    Updates the data in-place and returns a dict of variables:removed categories
    """
    columns = _validate_skip_only(data, skip, only)
    dtypes = data.loc[:, columns].dtypes
    catvars = [v for v in dtypes[dtypes == 'category'].index]
    removed_cats = dict()
    for var in catvars:
        counts = data[var].value_counts()
        keep_cats = list(counts[counts > 0].index)
        if len(keep_cats) < len(counts):
            removed_cats[var] = set(counts.index) - set(keep_cats)
            data[var].cat.set_categories(new_categories=keep_cats,
                                         ordered=data[var].cat.ordered,
                                         inplace=True)
    return removed_cats


def validate_ewas_params(covariates, data, phenotype, survey_design_spec):
    # Covariates must be a list
    if type(covariates) != list:
        raise ValueError("'covariates' must be specified as a list.  Use an empty list ([]) if there aren't any.")

    # Make sure the index of each dataset is not a multiindex and give it a consistent name
    if isinstance(data.index, pd.MultiIndex):
        raise ValueError("Data must not have a multiindex")
    data.index.name = "ID"

    # Collects lists of regression variables
    types = _get_dtypes(data)
    rv_bin = [v for v, t in types.iteritems() if t == 'binary' and v not in covariates and v != phenotype]
    rv_cat = [v for v, t in types.iteritems() if t == 'categorical' and v not in covariates and v != phenotype]
    rv_cont = [v for v, t in types.iteritems() if t == 'continuous' and v not in covariates and v != phenotype]
    # Ensure there are variables which can be regressed
    if len(rv_bin + rv_cat + rv_cont) == 0:
        raise ValueError("No variables are available to run regression on")
    else:
        click.echo(
            f"Running {len(rv_bin):,} binary, {len(rv_cat):,} categorical, and {len(rv_cont):,} continuous variables")
    # Ensure covariates are all present and not unknown type
    covariate_types = [types.get(c, None) for c in covariates]
    missing_covariates = [c for c, dt in zip(covariates, covariate_types) if dt is None]
    unknown_covariates = [c for c, dt in zip(covariates, covariate_types) if dt == 'unknown']
    if len(missing_covariates) > 0:
        raise ValueError(f"One or more covariates were not found in the data: {', '.join(missing_covariates)}")
    if len(unknown_covariates) > 0:
        raise ValueError(f"One or more covariates have an unknown datatype: {', '.join(unknown_covariates)}")

    # Validate the type of the phenotype variable
    pheno_kind = types.get(phenotype, None)
    if phenotype in covariates:
        raise ValueError(f"The phenotype ('{phenotype}') cannot also be a covariate.")
    elif pheno_kind is None:
        raise ValueError(f"The phenotype ('{phenotype}') was not found in the data.")
    elif pheno_kind == 'unknown':
        raise ValueError(f"The phenotype ('{phenotype}') has an unknown type.")
    elif pheno_kind == 'constant':
        raise ValueError(f"The phenotype ('{phenotype}') is a constant value.")
    elif pheno_kind == 'categorical':
        raise NotImplementedError("Categorical Phenotypes are not yet supported.")
    elif pheno_kind == 'continuous':
        click.echo("Running EWAS on a Continuous Outcome (family = Gaussian)")
    elif pheno_kind == 'binary':
        # Use the order according to the categorical
        counts = data[phenotype].value_counts().to_dict()
        categories = data[phenotype].cat.categories
        codes, categories = zip(*enumerate(categories))
        data[phenotype].replace(categories, codes, inplace=True)
        click.echo(click.style(f"Running EWAS on a Binary Outcome (family = Binomial)\n"
                               f"\t{counts[categories[0]]:,} occurrences of '{categories[0]}' coded as 0\n"
                               f"\t{counts[categories[1]]:,} occurrences of '{categories[1]}' coded as 1",
                               fg='green'))
    else:
        raise ValueError("The phenotype's type could not be determined.  Please report this error.")

    # Log how many NA outcomes
    na_outcome_count = data[phenotype].isna().sum()
    click.echo(click.style(f"Using {len(data) - na_outcome_count:,} of {len(data):,} observations", fg="green"))
    if na_outcome_count > 0:
        click.echo(click.style(f"\t{na_outcome_count:,} are missing a value for the outcome variable", fg="green"))

    # Log Survey Design if it is being used
    if survey_design_spec is not None:
        click.echo(click.style(f"Using a Survey Design:\n{survey_design_spec}", fg='green'))

    return rv_bin, rv_cat, rv_cont, pheno_kind
