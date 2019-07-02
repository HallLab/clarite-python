import click
from ...modules import analyze


@click.group(name='analyze')
def analyze_cli():
    pass


@analyze_cli.command()
def ewas():
    """Run EWAS"""
    print("Ran EWAS")
    pass


@analyze_cli.command()
def add_corrected_pvalues():
    """"""
    pass
