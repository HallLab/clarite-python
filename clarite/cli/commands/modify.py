import click
from ...modules import modify


@click.group(name='modify')
def modify_cli():
    pass


@modify_cli.command()
def colfilter_percent_zero():
    pass


@modify_cli.command()
def colfilter_min_n():
    pass


@modify_cli.command()
def colfilter_min_cat_n():
    pass


@modify_cli.command()
def rowfilter_incomplete_observations():
    pass


@modify_cli.command()
def recode_values():
    pass


@modify_cli.command()
def remove_outliers():
    pass


@modify_cli.command()
def make_binary():
    pass


@modify_cli.command()
def make_categorical():
    pass


@modify_cli.command()
def make_continuous():
    pass


@modify_cli.command()
def merge_variables():
    pass
