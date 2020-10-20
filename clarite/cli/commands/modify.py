import click

from ...modules import modify
from ..parameters import (
    arg_data,
    arg_output,
    option_skip,
    option_only,
    CLARITE_DATA,
    OUTPUT_FILE,
)
from ..custom_types import ClariteData, save_clarite_data


@click.group(name="modify")
def modify_cli():
    pass


@modify_cli.command(help="Remove some variables from a dataset")
@arg_data
@arg_output
@option_skip
@option_only
def colfilter(data, output, skip, only):
    """Load Data, remove some variables, and save the data"""
    data.df = modify.colfilter(data.df, skip=skip, only=only)
    # Save
    save_clarite_data(data, output)


@modify_cli.command(
    help="Filter variables based on the fraction of observations with a value of zero"
)
@arg_data
@arg_output
@click.option(
    "-p",
    "--filter-percent",
    default="90.0",
    type=click.FloatRange(min=0, max=100),
    help="Remove variables when the percentage of observations equal to 0 is >= this value (0 to 100)",
)
@option_skip
@option_only
def colfilter_percent_zero(data, output, filter_percent, skip, only):
    # Modify
    data.df = modify.colfilter_percent_zero(
        data=data.df, filter_percent=filter_percent, skip=skip, only=only
    )
    # Save
    save_clarite_data(data, output)


@modify_cli.command(
    help="Filter variables based on a minimum number of non-NA observations per category"
)
@arg_data
@arg_output
@click.option(
    "-n",
    default=200,
    type=click.IntRange(min=0),
    help="Remove variables with less than this many non-na observations in each category",
)
@option_skip
@option_only
def colfilter_min_cat_n(data, output, n, skip, only):
    # Modify
    data.df = modify.colfilter_min_cat_n(data=data.df, n=n, skip=skip, only=only)
    # Save
    save_clarite_data(data, output)


@modify_cli.command(
    help="Filter variables based on a minimum number of non-NA observations"
)
@arg_data
@arg_output
@click.option(
    "-n",
    default=200,
    type=click.IntRange(min=0),
    help="Remove variables with less than this many non-na observations",
)
@option_skip
@option_only
def colfilter_min_n(data, output, n, skip, only):
    # Modify
    data.df = modify.colfilter_min_n(data=data.df, n=n, skip=skip, only=only)
    # Save
    save_clarite_data(data, output)


@modify_cli.command(
    help="Replace values in the data with other values."
    "The value being replaced ('current') and the new value ('replacement') "
    "are specified with their type, and only one may be included for each. "
    "If it is not specified, the value being replaced or being inserted is None."
)
@arg_data
@arg_output
@click.option(
    "--current-str",
    "cs",
    type=click.STRING,
    default=None,
    help="Replace occurences of this string value",
)
@click.option(
    "--current-int",
    "ci",
    type=click.INT,
    default=None,
    help="Replace occurences of this integer value",
)
@click.option(
    "--current-float",
    "cf",
    type=click.FLOAT,
    default=None,
    help="Replace occurences of this float value",
)
@click.option(
    "--replacement-str",
    "rs",
    type=click.STRING,
    default=None,
    help="Insert this string value",
)
@click.option(
    "--replacement-int",
    "ri",
    type=click.INT,
    default=None,
    help="Insert this integer value",
)
@click.option(
    "--replacement-float",
    "rf",
    type=click.FLOAT,
    default=None,
    help="Insert this float value",
)
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
        raise ValueError(
            "The 'current' value was specified for multiple types.  It should be specified with at most one type."
        )
    # Decode replacement
    r = [v for v in (rs, ri, rf) if v is not None]
    if len(r) == 0:
        replacement = None
    elif len(r) == 1:
        replacement = r[0]
    else:
        raise ValueError(
            "The 'replacement' value was specified for multiple types.  It should be specified with at most one type."
        )
    # Modify
    data.df = modify.recode_values(
        data=data.df, replacement_dict={current: replacement}, skip=skip, only=only
    )
    # Save
    save_clarite_data(data, output)


@modify_cli.command(
    help="Replace outlier values with NaN.  Outliers are defined using a gaussian or IQR approach."
)
@arg_data
@arg_output
@click.option(
    "--method", "-m", type=click.Choice(["gaussian", "iqr"]), default="gaussian"
)
@click.option("--cutoff", "-c", type=click.FLOAT, default=3.0)
@option_skip
@option_only
def remove_outliers(data, output, method, cutoff, skip, only):
    # Modify
    data.df = modify.remove_outliers(
        data=data.df, method=method, cutoff=cutoff, skip=skip, only=only
    )
    # Save
    save_clarite_data(data, output)


@modify_cli.command(
    help="Select some rows from a dataset using a simple comparison, keeping rows where the comparison is True."
)
@arg_data
@arg_output
@click.argument("column", type=click.STRING)
@click.option(
    "--value-str",
    "vs",
    type=click.STRING,
    default=None,
    help="Compare values in the column to this string",
)
@click.option(
    "--value-int",
    "vi",
    type=click.INT,
    default=None,
    help="Compare values in the column to this integer",
)
@click.option(
    "--value-float",
    "vf",
    type=click.FLOAT,
    default=None,
    help="Compare values in the column to this floating point number",
)
@click.option(
    "--comparison",
    "-c",
    default="eq",
    type=click.Choice(["lt", "lte", "eq", "gte", "gt"]),
    help="Keep rows where the value of the column is lt (<), lte (<=), eq (==), gte (>=), or gt (>) the specified value.  Eq by default.",
)
def rowfilter(data, output, column, vs, vi, vf, comparison):
    """Load Data, keep certain rows, and save the data"""
    # Ensure column is present
    if column not in data.df.columns:
        raise ValueError(f"The specified column {column} was not found in the data.")
    # Decode value
    values = [v for v in (vs, vi, vf) if v is not None]
    if len(values) != 1:
        raise ValueError(
            "The comparison value ('--value-str', '--value-int', or '--value-float') must be specified just once."
        )
    else:
        value = values[0]
    # Filter
    if comparison == "lt":
        data.df = data.df.loc[
            data.df[column] < value,
        ]
    elif comparison == "lte":
        data.df = data.df.loc[
            data.df[column] <= value,
        ]
    elif comparison == "eq":
        data.df = data.df.loc[
            data.df[column] == value,
        ]
    elif comparison == "gt":
        data.df = data.df.loc[
            data.df[column] >= value,
        ]
    elif comparison == "gte":
        data.df = data.df.loc[
            data.df[column] > value,
        ]
    # Save
    save_clarite_data(data, output)


@modify_cli.command(
    help="Filter out observations that are not complete cases (contain no NA values)"
)
@arg_data
@arg_output
@option_skip
@option_only
def rowfilter_incomplete_obs(data, output, skip, only):
    # Modify
    data.df = modify.rowfilter_incomplete_obs(data=data.df, skip=skip, only=only)
    # Save
    save_clarite_data(data, output)


@modify_cli.command(help="Set the type of variables to 'binary'")
@arg_data
@arg_output
@option_skip
@option_only
def make_binary(data, output, skip, only):
    # Modify
    data.df = modify.make_binary(data=data.df, skip=skip, only=only)
    # Save
    save_clarite_data(data, output)


@modify_cli.command(help="Set the type of variables to 'categorical'")
@arg_data
@arg_output
@option_skip
@option_only
def make_categorical(data, output, skip, only):
    # Modify
    data.df = modify.make_categorical(data=data.df, skip=skip, only=only)
    # Save
    save_clarite_data(data, output)


@modify_cli.command(help="Set the type of variables to 'continuous'")
@arg_data
@arg_output
@option_skip
@option_only
def make_continuous(data, output, skip, only):
    # Modify
    data.df = modify.make_continuous(data=data.df, skip=skip, only=only)
    # Save
    save_clarite_data(data, output)


@modify_cli.command(help="Apply a function to each value of a variable")
@arg_data
@arg_output
@click.argument("transform_method", type=click.STRING)
@option_skip
@option_only
def transform_variable(data, output, transform_method, skip, only):
    # Create new variable
    data.df = modify.transform(
        data=data.df, transform_method=transform_method, skip=skip, only=only
    )
    # Save
    save_clarite_data(data, output)


@modify_cli.command(help="Remove extra categories from categorical datatypes")
@arg_data
@arg_output
@option_skip
@option_only
def drop_extra_categories(data, output, skip, only):
    # Create new variable
    data.df = modify.drop_extra_categories(data=data.df, skip=skip, only=only)
    # Save
    save_clarite_data(data, output)


@modify_cli.command(help="Categorize data based on the number of unique values")
@arg_data
@arg_output
@click.option(
    "--cat_min",
    default=3,
    help="Minimum number of unique values in a variable to make it a categorical type",
)
@click.option(
    "--cat_max",
    default=6,
    help="Maximum number of unique values in a variable to make it a categorical type",
)
@click.option(
    "--cont_min",
    default=15,
    help="Minimum number of unique values in a variable to make it a continuous type",
)
def categorize(data, output, cat_min, cat_max, cont_min):
    # Categorize and convert to ClariteData types
    data.df = modify.categorize(
        data.df, cat_min=cat_min, cat_max=cat_max, cont_min=cont_min
    )
    # Save Data
    save_clarite_data(data=data, output=output)


@modify_cli.command(help="Merge variables from two different datasets into one")
@click.argument("left", type=CLARITE_DATA)
@click.argument("right", type=CLARITE_DATA)
@arg_output
@click.option(
    "--how",
    "-h",
    default="outer",
    type=click.Choice(["left", "right", "inner", "outer"]),
    help="Type of Merge",
)
def merge_variables(left, right, output, how):
    # Merge
    result = modify.merge_variables(left.df, right.df, how)
    result = ClariteData(name=f"{left.name}.{right.name}", df=result)
    # Save
    save_clarite_data(result, output)


@modify_cli.command(help="Merge observations from two different datasets into one")
@click.argument("top", type=CLARITE_DATA)
@click.argument("bottom", type=CLARITE_DATA)
@arg_output
def merge_observations(top, bottom, output):
    # Merge
    result = modify.merge_observations(top.df, bottom.df)
    result = ClariteData(name=f"{top.name}.{bottom.name}", df=result)
    # Save
    save_clarite_data(result, output)


@modify_cli.command(help="Move variables from one dataset to another")
@click.argument("left", type=CLARITE_DATA)
@click.argument("right", type=CLARITE_DATA)
@click.option("--output_left", default=None, type=OUTPUT_FILE)
@click.option("--output_right", default=None, type=OUTPUT_FILE)
@option_skip
@option_only
def move_variables(left, right, output_left, output_right, skip, only):
    # Move
    left_df, right_df = modify.move_variables(left.df, right.df, skip=skip, only=only)
    # Save
    # Modify in-place if output is not specified
    if output_left is None:
        output_left = left.name
    if output_right is None:
        output_right = right.name
    left = ClariteData(name=left.name, df=left_df)
    right = ClariteData(name=right.name, df=right_df)
    save_clarite_data(left, output_left)
    save_clarite_data(right, output_right)
