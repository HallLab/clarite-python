import click
from ...modules import io


@click.group(name='io')
def io_cli():
    pass


@io_cli.command()
def load_data():
    """Load Data"""
    print("Loaded Data")
    pass


@io_cli.command()
def save():
    """Save Data"""
    print("Saved Data")
    pass


@io_cli.command()
def save_dtypes():
    pass
