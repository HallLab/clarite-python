import click
from ...modules import plot


@click.group(name='plot')
def plot_cli():
    pass


@plot_cli.command()
def histogram():
    pass


@plot_cli.command()
def distributions():
    pass


@plot_cli.command()
def manhattan():
    pass
