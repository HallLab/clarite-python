import click
from ...modules import modify
from ...internal.utilities import _validate_skip_only
from ..parameters import arg_data, option_output, option_skip, option_only
from ..custom_types import save_clarite_data


@click.group(name='modify')
def modify_cli():
    pass


@modify_cli.command(help="Remove some columns from a dataset")
@arg_data
@option_output
@option_skip
@option_only
def colfilter(data, output, skip, only):
    """Load Data, remove some columns, and save the data"""
    # Process skip/only parameters
    columns = _validate_skip_only(list(data.df), skip, only)
    data.df = data.df[columns]
    # Save
    save_clarite_data(data, output)


@modify_cli.command(help="Filter variables based on the fraction of observations with a value of zero")
@arg_data
@option_output
@click.option('-p', '--filter-percent', default='90.0', type=click.FloatRange(min=0, max=100),
              help="Remove variables when the percentage of observations equal to 0 is >= this value (0 to 100)")
@option_skip
@option_only
def colfilter_percent_zero(data, output, filter_percent, skip, only):
    # Modify
    data.df = modify.colfilter_percent_zero(data=data.df, filter_percent=filter_percent, skip=skip, only=only)
    # Save
    save_clarite_data(data, output)


@modify_cli.command(help="Filter variables based on a minimum number of non-NA observations")
@arg_data
@option_output
@click.option('-n', default=200, type=click.IntRange(min=0),
              help="Remove variables with less than this many non-na observations")
@option_skip
@option_only
def colfilter_min_n(data, output, n, skip, only):
    # Modify
    data.df = modify.colfilter_min_n(data=data.df, n=n, skip=skip, only=only)
    # Save
    save_clarite_data(data, output)


@modify_cli.command(help="Filter variables based on a minimum number of non-NA observations per category")
@arg_data
@option_output
@click.option('-n', default=200, type=click.IntRange(min=0),
              help="Remove variables with less than this many non-na observations in each category")
@option_skip
@option_only
def colfilter_min_cat_n(data, output, n, skip, only):
    # Modify
    data.df = modify.colfilter_min_cat_n(data=data.df, n=n, skip=skip, only=only)
    # Save
    save_clarite_data(data, output)


@modify_cli.command(help="Select some rows from a dataset using a simple comparison, keeping rows where the comparison is True.")
@arg_data
@option_output
@click.argument('column', type=click.STRING)
@click.option('--value-str', 'vs', type=click.STRING, default=None, help="Compare values in the column to this string")
@click.option('--value-int', 'vi', type=click.INT, default=None, help="Compare values in the column to this integer")
@click.option('--value-float', 'vf', type=click.FLOAT, default=None, help="Compare values in the column to this floating point number")
@click.option('--comparison', '-c', default='eq', type=click.Choice(['lt', 'lte', 'eq', 'gte', 'gt']),
              help="Keep rows where the value of the column is lt (<), lte (<=), eq (==), gte (>=), or gt (>) the specified value.  Eq by default.")
def rowfilter(data, output, column, vs, vi, vf, comparison):
    """Load Data, keep certain rows, and save the data"""
    # Ensure column is present
    if column not in data.df.columns:
        raise ValueError(f"The specified column {column} was not found in the data.")
    # Decode value
    values = [v for v in (vs, vi, vf) if v is not None]
    if len(values) != 1:
        raise ValueError("The comparison value ('--value-str', '--value-int', or '--value-float') must be specified just once.")
    else:
        value = values[0]
    # Filter
    if comparison == 'lt':
        data.df = data.df.loc[data.df[column] < value, ]
    elif comparison == 'lte':
        data.df = data.df.loc[data.df[column] <= value, ]
    elif comparison == 'eq':
        data.df = data.df.loc[data.df[column] == value, ]
    elif comparison == 'gt':
        data.df = data.df.loc[data.df[column] >= value, ]
    elif comparison == 'gte':
        data.df = data.df.loc[data.df[column] > value, ]
    # Save
    save_clarite_data(data, output)


@modify_cli.command(help="Filter out observations that are not complete cases (contain no NA values)")
@arg_data
@option_output
@option_skip
@option_only
def rowfilter_incomplete_obs(data, output, skip, only):
    # Modify
    data.df = modify.rowfilter_incomplete_obs(data=data.df, skip=skip, only=only)
    # Save
    save_clarite_data(data, output)


@modify_cli.command(help="Replace values in the data with other values."
                         "The value being replaced ('current') and the new value ('replacement') "
                         "are specified with their type, and only one may be included for each. "
                         "If it is not specified, the value being replaced or being inserted is None.")
@arg_data
@option_output
@click.option('--current-str', 'cs', type=click.STRING, default=None, help="Replace occurences of this string value")
@click.option('--current-int', 'ci', type=click.INT, default=None, help="Replace occurences of this integer value")
@click.option('--current-float', 'cf', type=click.FLOAT, default=None, help="Replace occurences of this float value")
@click.option('--replacement-str', 'rs', type=click.STRING, default=None, help="Insert this string value")
@click.option('--replacement-int', 'ri', type=click.INT, default=None, help="Insert this integer value")
@click.option('--replacement-float', 'rf', type=click.FLOAT, default=None, help="Insert this float value")
@option_skip
@option_only
def recode_values(data, output, cs, ci, cf, rs, ri, rf, skip, only):
    # Decode current
    c = [v for v in (cs, ci, cf) if v is not None]
    if len(c) == 0:
        current = None
    elif len(c) == 1:
        current = c[0]
    else:
        raise ValueError("The 'current' value was specified for multiple types.  It should be specified with at most one type.")
    # Decode replacement
    r = [v for v in (rs, ri, rf) if v is not None]
    if len(r) == 0:
        replacement = None
    elif len(r) == 1:
        replacement = r[0]
    else:
        raise ValueError("The 'replacement' value was specified for multiple types.  It should be specified with at most one type.")
    # Modify
    data.df = modify.recode_values(data=data.df, replacement_dict={current: replacement}, skip=skip, only=only)
    # Save
    save_clarite_data(data, output)


@modify_cli.command(help="Replace outlier values with NaN.  Outliers are defined using a gaussian or IQR approach.")
@arg_data
@option_output
@click.option('--method', '-m', type=click.Choice(['gaussian', 'iqr']), default='gaussian')
@click.option('--cutoff', '-c', type=click.FLOAT, default=3.0)
@option_skip
@option_only
def remove_outliers(data, output, method, cutoff, skip, only):
    # Modify
    data.df = modify.remove_outliers(data=data.df, method=method, cutoff=cutoff, skip=skip, only=only)
    # Save
    save_clarite_data(data, output)


@modify_cli.command(help="Set the type of variables to 'binary'")
@arg_data
@option_output
@option_skip
@option_only
def make_binary(data, output, skip, only):
    # Modify
    data.df = modify.make_binary(data=data.df, skip=skip, only=only)
    # Save
    save_clarite_data(data, output)


@modify_cli.command(help="Set the type of variables to 'categorical'")
@arg_data
@option_output
@option_skip
@option_only
def make_categorical(data, output, skip, only):
    # Modify
    data.df = modify.make_categorical(data=data.df, skip=skip, only=only)
    # Save
    save_clarite_data(data, output)


@modify_cli.command(help="Set the type of variables to 'continuous'")
@arg_data
@option_output
@option_skip
@option_only
def make_continuous(data, output, skip, only):
    # Modify
    data.df = modify.make_continuous(data=data.df, skip=skip, only=only)
    # Save
    save_clarite_data(data, output)


@modify_cli.command(help="Apply a function to each value of a variable")
@arg_data
@option_output
@click.argument('variable', type=click.STRING)
@click.argument('transform', type=click.STRING)
@click.argument('new_name', type=click.STRING)
def transform_variable(data, output, variable, transform, new_name):
    # Create new variable
    try:
        data.df[new_name] = data.df[variable].apply(transform)
    except Exception:
        raise ValueError(f"Couldn't apply a function named '{transform}'' to '{variable}'' in order to create a new '{new_name}' variable.")
    # Drop old variable
    if new_name != variable:
        data.df = data.df.drop(variable, axis='columns')
    # Save
    save_clarite_data(data, output)
