from pathlib import Path

import click
import pandas as pd
from matplotlib import pyplot as plt

from ...modules import plot, io
from ...modules.analyze import result_columns, corrected_pvalue_columns
from ..parameters import input_file, output_file


@click.group(name='plot')
def plot_cli():
    pass


@plot_cli.command(help="Create a histogram plot of a variable")
@click.argument('data', type=input_file)
@click.argument('variable', type=click.STRING)
@click.argument('output', type=output_file)
def histogram(data, variable, output):
    # Load data
    data = io.load_data(data)
    # Plot
    plot.histogram(data=data, column=variable)
    # Save and Close
    plt.savefig(output)
    plt.close()
    # Log
    click.echo(click.style(f"Done: Saved plot to {output}", fg='green'))


@plot_cli.command(help="Generate a pdf containing distribution plots for each variable")
@click.argument('data', type=input_file)
@click.argument('output', type=output_file)
@click.option('--kind', '-k', default='count', type=click.Choice(['count', 'box', 'violin', 'qq']),
              help="Kind of plot used for continuous data.  Non-continuous always shows a count plot.")
@click.option('--nrows', default=4, type=click.IntRange(min=1, max=10), help="Number of rows per page")
@click.option('--ncols', default=3, type=click.IntRange(min=1, max=10), help="Number of columns per page")
@click.option('--quality', '-q', default='medium', type=click.Choice(['low', 'medium', 'high']),
              help="Quality of the generated plots: low (150 dpi), medium (300 dpi), or high (1200 dpi).")
@click.option('--sort/--no-sort', help="Sort variables alphabetically")
def distributions(data, output, kind, nrows, ncols, quality, sort):
    # Load data
    data = io.load_data(data)
    # Plot and save
    plot.distributions(data=data, filename=output, continuous_kind=kind, nrows=nrows, ncols=ncols, quality=quality, sort=sort)
    # Log
    click.echo(click.style(f"Done: Saved plot to {output}", fg='green'))


@plot_cli.command(help="Generate a manhattan plot of EWAS results")
@click.argument('data', type=input_file)
@click.argument('output', type=output_file)
@click.option('--categories', '-c', type=input_file, default=None, help="tab-separate file with two columns: 'Variable' and 'category'")
@click.option('--other', '-o', multiple=True, help="other datasets to include in the plot")
@click.option('--nlabeled', default=3, type=click.IntRange(min=0, max=50), help="label top n points")
@click.option('--label', default=None, multiple=True, type=click.STRING, help="label points by name")
def manhattan(data, output, categories, other, nlabeled, label):
    # Load data
    data = {Path(data).name: pd.read_csv(data, sep="\t", index_col=['variable', 'phenotype'])}
    for o in other:
        data[Path(o).name] = pd.read_csv(o, sep="\t", index_col=['variable', 'phenotype'])
    for d_name, d in data.items():
        if list(d) != result_columns + corrected_pvalue_columns:
            raise ValueError(f"{d_name} was not a valid EWAS result file.")
    # Load categories, if any
    if categories is not None:
        categories = pd.read_csv(categories, sep="\t")
        categories.columns = ['Variable', 'category']
        categories = categories.set_index('Variable')['category'].to_dict()
    # Plot and save
    plot.manhattan(data, categories=categories, num_labeled=nlabeled, label_vars=label, filename=output)
    # Log
    click.echo(click.style(f"Done: Saved plot to {output}", fg='green'))
