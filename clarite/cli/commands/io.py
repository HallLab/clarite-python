import click
from ...modules import io
from ..parameters import input_file, output_file


@click.group(name='io')
def io_cli():
    pass


@io_cli.command(help="Load data from a tab-separated or comma-separated file and save it in the standard format")
@click.argument('data', type=input_file)
@click.argument('output', type=output_file)
@click.option('--index', '-i', type=click.STRING, help="Name of the column to use as the index.  Default is the first column.")
@click.option('--tab', is_flag=True, help="File is tab-separated")
@click.option('--comma', is_flag=True, help="File is comma-separated")
def load_data(data, output, index, tab, comma):
    """Load Data from some format and save it in the standard format for further processing"""
    # Determine seperator character
    if tab and comma:
        raise ValueError(f"Can't specify '--tab' and '--comma' at the same time")
    elif not tab and not comma:
        raise ValueError(f"Must specify either '--tab' or '--comma'")
    elif tab and not comma:
        sep = "\t"
    elif comma and not tab:
        sep = ","
    # Index column is assumed to be the first column if not provided
    if index is None:
        index = 0
    # Load Data, ignoring any existing dtypes file
    data = io.load_data(filename=data, index_col=index, sep=sep, dtypes=False)
    # Raise error if no variables were found (only one column)
    if len(data.columns) == 0:
        raise ValueError("Only one column was found- was the correct delimeter used?")
    # Save
    data.to_csv(output, sep="\t")
    # Log
    click.echo(click.style(f"Done: Saved {len(data.columns):,} variables with {len(data):,} observations to {output}", fg='green'))
