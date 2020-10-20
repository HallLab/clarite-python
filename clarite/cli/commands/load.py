import click
from ...modules import load
from ...internal.utilities import _validate_skip_only
from ..parameters import INPUT_FILE, arg_output, option_skip, option_only
from ..custom_types import ClariteData, save_clarite_data


@click.group(name="load")
def load_cli():
    pass


@load_cli.command(
    help="Load data from a tab-separated file and save it in the standard format"
)
@click.argument("input", type=INPUT_FILE)
@arg_output
@click.option(
    "--index",
    "-i",
    type=click.STRING,
    help="Name of the column to use as the index.  Default is the first column.",
)
@option_skip
@option_only
def from_tsv(input, output, index, skip, only):
    """Load Data from a tab-separated file and save it in the standard format for further processing"""
    # Index column is assumed to be the first column if not provided
    if index is None:
        index = 0
    # Load Data, ignoring any existing dtypes file
    data = load.from_tsv(filename=input, index_col=index)
    # Raise error if no variables were found (only one column)
    if len(data.columns) == 0:
        raise ValueError("Only one column was found- was the correct delimeter used?")
    # Process skip/only parameters
    columns = _validate_skip_only(data, skip, only)
    data = data.loc[:, columns]
    # Convert to a ClariteData object and save
    data = ClariteData(name=input, df=data)
    save_clarite_data(data, output)


@load_cli.command(
    help="Load data from a comma-separated file and save it in the standard format"
)
@click.argument("input", type=INPUT_FILE)
@arg_output
@click.option(
    "--index",
    "-i",
    type=click.STRING,
    help="Name of the column to use as the index.  Default is the first column.",
)
@option_skip
@option_only
def from_csv(input, output, index, skip, only):
    """Load Data from a tab-separated file and save it in the standard format for further processing"""
    # Index column is assumed to be the first column if not provided
    if index is None:
        index = 0
    # Load Data, ignoring any existing dtypes file
    data = load.from_csv(filename=input, index_col=index)
    # Raise error if no variables were found (only one column)
    if len(data.columns) == 0:
        raise ValueError("Only one column was found- was the correct delimeter used?")
    # Process skip/only parameters
    columns = _validate_skip_only(data, skip, only)
    data = data.loc[:, columns]
    # Convert to a ClariteData object and save
    data = ClariteData(name=input, df=data)
    save_clarite_data(data, output)
