import click
from ...modules import load
from ...internal.utilities import _validate_skip_only
from ..parameters import INPUT_FILE, OUTPUT_FILE, skip, only
from ..custom_types import ClariteData


@click.group(name='load')
def load_cli():
    pass


@load_cli.command(help="Load data from a tab-separated file and save it in the standard format")
@click.argument('input', type=INPUT_FILE)
@click.argument('output', type=OUTPUT_FILE)
@click.option('--index', '-i', type=click.STRING, help="Name of the column to use as the index.  Default is the first column.")
@skip
@only
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
    columns = _validate_skip_only(list(data), skip, only)
    data = data[columns]
    # Convert to a ClariteData object and save
    # Save
    data = ClariteData(name=input, output=output, df=data)
    data.save_data()
