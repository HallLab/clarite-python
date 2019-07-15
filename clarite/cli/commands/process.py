import click
import pandas as pd
from ...modules import process
from ..parameters import clarite_data_arg, CLARITE_DATA, INPUT_FILE, OUTPUT_FILE, skip, only
from ..custom_types import ClariteData


@click.group(name='process')
def process_cli():
    pass


@process_cli.command(help="Categorize data based on the number of unique values")
@clarite_data_arg
@click.argument('output', type=OUTPUT_FILE)
@click.option('--cat_min', default=3, help="Minimum number of unique values in a variable to make it a categorical type")
@click.option('--cat_max', default=6, help="Maximum number of unique values in a variable to make it a categorical type")
@click.option('--cont_min', default=15, help="Minimum number of unique values in a variable to make it a continuous type")
def categorize(data, output, cat_min, cat_max, cont_min):
    # Get output
    output = data.output
    # Categorize and convert to ClariteData types
    df_bin, df_cat, df_cont, df_check = process.categorize(data.df, cat_min=cat_min, cat_max=cat_max, cont_min=cont_min)
    df_bin = ClariteData(data.name + "_bin", output=str(output.with_suffix('')) + f"_bin", df=df_bin)
    df_cat = ClariteData(data.name + "_cat", output=str(output.with_suffix('')) + f"_cat", df=df_cat)
    df_cont = ClariteData(data.name + "_cont", output=str(output.with_suffix('')) + f"_cont", df=df_cont)
    df_check = ClariteData(data.name + "_check", output=str(output.with_suffix('')) + f"_check", df=df_check)
    # Save Data
    df_bin.save_data()
    df_cat.save_data()
    df_cont.save_data()
    df_check.save_data()


@process_cli.command(help="Merge variables from two different datasets into one")
@click.argument('left', type=CLARITE_DATA)
@click.argument('right', type=CLARITE_DATA)
@click.argument('output', type=OUTPUT_FILE)
@click.option('--how', '-h', default='outer', type=click.Choice(['left', 'right', 'inner', 'outer']), help="Type of Merge")
def merge_variables(left, right, output, how):
    # Merge
    result = process.merge_variables(left.df, right.df, how)
    result = ClariteData(name=f"{left.name}.{right.name}", output=output, df=result)
    # Save
    result.save_data()


@process_cli.command(help="Merge rows from two different datasets into one")
@click.argument('top', type=CLARITE_DATA)
@click.argument('bottom', type=CLARITE_DATA)
@click.argument('output', type=OUTPUT_FILE)
def merge_rows(top, bottom, output):
    # Merge
    extra = set(list(top.df)) - set(list(bottom.df))
    missing = set(list(bottom.df)) - set(list(top.df))
    if len(extra) > 0:
        raise ValueError(f"Couldn't merge rows: Extra columns in the 'bottom' data: {', '.join(extra)}")
    elif len(missing) > 0:
        raise ValueError(f"Couldn't merge rows: Missing columns in the 'bottom' data: {', '.join(missing)}")
    elif (top.dtypes != bottom.dtypes).any():
        raise ValueError("Couldn't merge rows: different data types")
    else:
        result = pd.concat([top.df, bottom.df], verify_integrity=True, sort=False)
        result = ClariteData(name=f"{top.name}.{bottom.name}", output=output, df=result)
    # Save
    result.save_data()


@process_cli.command(help="Move variables from one dataset to another")
@click.argument('left', type=INPUT_FILE)
@click.argument('right', type=INPUT_FILE)
@click.argument('output_left', type=OUTPUT_FILE)
@click.argument('output_right', type=OUTPUT_FILE)
@skip
@only
def move_variables(left, right, output_left, output_right, skip, only):
    # Move
    left_df, right_df = process.move_variables(left.df, right.df, skip=skip, only=only)
    # Save
    left = ClariteData(name=left.name, output=output_left, df=left_df)
    right = ClariteData(name=right.name, output=output_right, df=right_df)
    left.save_data()
    right.save_data()
