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


@plot_cli.command()
def distributions():
    pass


@plot_cli.command()
def manhattan():
    pass
