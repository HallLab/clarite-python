import click
from ...modules import modify, io
from ..parameters import input_file, output_file, skip, only


@click.group(name='modify')
def modify_cli():
    pass


@modify_cli.command()
@click.argument('data', type=input_file)
@click.argument('output', type=output_file)
@click.option('-p', '--filter-percent', default='90.0', type=click.FloatRange(min=0, max=100))
@skip
@only
def colfilter_percent_zero(data, output, filter_percent, skip, only):
    # Load data
    data = io.load_data(data)
    # Modify
    result = modify.colfilter_percent_zero(data=data, filter_percent=filter_percent, skip=skip, only=only)
    # Save
    result.to_csv(output, sep="\t")
    # Log
    click.echo(click.style(f"Done: Saved filtered data to {output}", fg='green'))


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
