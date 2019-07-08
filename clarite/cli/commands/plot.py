import click
from matplotlib import pyplot as plt
from ...modules import plot, io
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
    # Plot
    plot.distributions(data=data, filename=output, continuous_kind=kind, nrows=nrows, ncols=ncols, quality=quality, sort=sort)


@plot_cli.command()
def manhattan():
    pass
