import click
from ...modules import describe, io
from ..parameters import input_file, output_file


@click.group(name='describe')
def describe_cli():
    pass


@describe_cli.command()
@click.argument('data', type=input_file)
@click.argument('output', type=output_file)
@click.option('-t', '--threshold', default=0.75, help="Report correlations with R >= this value")
def correlations(data, output, threshold):
    # Load data
    data = io.load_data(data)
    # Describe
    results = describe.correlations(data, threshold)
    # Save results
    results.to_csv(output, sep="\t", index=False)
    # Log
    click.echo(click.style(f"Done: Saved {len(results):,} correlations to {output}", fg='green'))


@describe_cli.command()
def freq_table():
    pass


@describe_cli.command()
def percent_na():
    pass
