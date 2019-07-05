from pathlib import Path
import click
from ...modules import process, io
from ..parameters import input_file, output_file, skip, only


@click.group(name='process')
def process_cli():
    pass


@process_cli.command()
@click.argument('data', type=input_file)
@click.option('--output', '-o', type=output_file, default=None, help="Output will have '_bin.txt', '_cat.txt', '_cont.txt', and '_check.txt' suffixes added")
@click.option('--cat_min', default=3, help="Minimum number of unique values in a variable to make it a categorical type")
@click.option('--cat_max', default=6, help="Maximum number of unique values in a variable to make it a categorical type")
@click.option('--cont_min', default=15, help="Minimum number of unique values in a variable to make it a continuous type")
def categorize(data, output, cat_min, cat_max, cont_min):
    # Get output prefix
    if output is None:
        output = Path(data)
    else:
        output = Path(output)
    # Load data
    data = io.load_data(data)
    # Categorize
    df_bin, df_cat, df_cont, df_check = process.categorize(data, cat_min=cat_min, cat_max=cat_max, cont_min=cont_min)
    # Save Data
    for name, df in zip(['binary', 'categorical', 'continuous', 'check'], [df_bin, df_cat, df_cont, df_check]):
        output_name = str(output.with_suffix('')) + f"_{name}"
        io.save(df, filename=output_name)
        click.echo(click.style(f"Saved {name} results to {output_name}", fg='green'))


@process_cli.command()
@click.argument('left', type=input_file)
@click.argument('right', type=input_file)
@click.argument('output', type=output_file)
@click.option('--how', '-h', default='outer', type=click.Choice(['left', 'right', 'inner', 'outer']), help="Type of Merge")
def merge_variables(left, right, output, how):
    # Load data
    left = io.load_data(left)
    right = io.load_data(right)
    # Merge
    result = process.merge_variables(left, right, how)
    # Save
    result.to_csv(output, sep="\t")
    # Log
    click.echo(click.style(f"Done: Saved {len(result.columns):,} with {len(result):,} variables to {output}", fg='green'))


@process_cli.command()
@click.argument('left', type=input_file)
@click.argument('right', type=input_file)
@click.argument('output_left', type=output_file)
@click.argument('output_right', type=output_file)
@skip
@only
def move_variables(left, right, output_left, output_right, skip, only):
    # Load data
    left = io.load_data(left)
    right = io.load_data(right)
    before = len(list(left))
    # Move
    left, right = process.move_variables(left, right, skip=skip, only=only)
    after = len(list(left))
    # Save
    left.to_csv(output_left, sep="\t")
    right.to_csv(output_right, sep="\t")
    # Log
    click.echo(click.style(f"Done: Moved {before-after:,} variables and saved results to {output_left} and {output_right}", fg='green'))
