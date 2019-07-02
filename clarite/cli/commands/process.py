import click
from ...modules import process


@click.group(name='process')
def process_cli():
    pass


@process_cli.command()
def categorize():
    pass
