import click
from ...modules import describe, io
from ..parameters import input_file, output_file


@click.group(name='describe')
def describe_cli():
    pass


@describe_cli.command(help="Report top correlations between variables")
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


@describe_cli.command(help="Report the number of occurences of each value for each variable")
@click.argument('data', type=input_file)
@click.argument('output', type=output_file)
def freq_table(data, output):
    # Load data
    data = io.load_data(data)
    # Describe
    results = describe.freq_table(data)
    # Save results
    results.to_csv(output, sep="\t", index=False)
    # Log
    processed = results.loc[results['value'] != '<Non-Categorical Values>', ]
    if len(processed) > 0:
        num_values = processed[['variable', 'value']].nunique()
        num_variables = processed['variable'].nunique()
    else:
        num_values = 0
        num_variables = 0
    click.echo(click.style(f"Done: Saved {num_values:,} unique value counts for {num_variables:,} non-continuous variables to {output}", fg='green'))


@describe_cli.command(help="Report the percent of observations that are NA for each variable")
@click.argument('data', type=input_file)
@click.argument('output', type=output_file)
def percent_na(data, output):
    # Load data
    data = io.load_data(data)
    # Describe
    results = describe.percent_na(data)
    # Save results
    results.to_csv(output, sep="\t", index=False)
    # Log
    click.echo(click.style(f"Done: Saved results for {len(results):,} variables to {output}", fg='green'))
