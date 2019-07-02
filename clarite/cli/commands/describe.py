import click
from ...modules import describe


@click.group(name='describe')
def describe_cli():
    pass


@describe_cli.command()
def correlations():
    pass


@describe_cli.command()
def freq_table():
    pass


@describe_cli.command()
def percent_na():
    pass
