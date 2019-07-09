from pathlib import Path
import click
import pandas as pd
from ...modules import process, io
from ..parameters import input_file, output_file, skip, only


@click.group(name='process')
def process_cli():
    pass


@process_cli.command(help="Categorize data based on the number of unique values")
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
        if len(df.columns) > 0:
            io.save(df, filename=output_name)
            click.echo(click.style(f"Saved {name} results to {output_name}", fg='green'))
        else:
            click.echo(click.style(f"No {name} variables available to save to {output_name}", fg='yellow'))


@process_cli.command(help="Merge variables from two different datasets into one")
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
    io.save(result, filename=output)
    # Log
    click.echo(click.style(f"Done: Saved {len(result.columns):,} with {len(result):,} variables to {output}", fg='green'))


@process_cli.command(help="Merge rows from two different datasets into one")
@click.argument('top', type=input_file)
@click.argument('bottom', type=input_file)
@click.argument('output', type=output_file)
def merge_rows(top, bottom, output):
    # Load data
    top = io.load_data(top)
    bottom = io.load_data(bottom)
    # Merge
    extra = set(list(top)) - set(list(bottom))
    missing = set(list(bottom)) - set(list(top))
    if len(extra) > 0:
        raise ValueError(f"Couldn't merge rows: Extra columns in the 'bottom' data: {', '.join(extra)}")
    elif len(missing) > 0:
        raise ValueError(f"Couldn't merge rows: Missing columns in the 'bottom' data: {', '.join(missing)}")
    elif (top.dtypes != bottom.dtypes).any():
        raise ValueError("Couldn't merge rows: different data types")
    else:
        result = pd.concat([top, bottom], verify_integrity=True, sort=False)
    # Save
    io.save(result, filename=output)
    # Log
    click.echo(click.style(f"Done: Saved {len(result.columns):,} with {len(result):,} variables to {output}", fg='green'))


@process_cli.command(help="Move variables from one dataset to another")
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
    io.save(left, filename=output_left)
    io.save(right, filename=output_right)
    # Log
    click.echo(click.style(f"Done: Moved {before-after:,} variables and saved results to {output_left} and {output_right}", fg='green'))
