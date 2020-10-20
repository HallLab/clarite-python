import click
from ...modules import describe
from ..parameters import arg_data, arg_output


@click.group(name="describe")
def describe_cli():
    pass


@describe_cli.command(help="Report top correlations between variables")
@arg_data
@arg_output
@click.option(
    "-t", "--threshold", default=0.75, help="Report correlations with R >= this value"
)
def correlations(data, output, threshold):
    # Describe
    results = describe.correlations(data.df, threshold)
    # Save results
    results.to_csv(output, sep="\t", index=False)
    # Log
    click.echo(
        click.style(
            f"Done: Saved {len(results):,} correlations to {output}", fg="green"
        )
    )


@describe_cli.command(
    help="Report the number of occurences of each value for each variable"
)
@arg_data
@arg_output
def freq_table(data, output):
    # Describe
    results = describe.freq_table(data.df)
    # Save results
    results.to_csv(output, sep="\t", index=False)
    # Log
    processed = results.loc[
        results["value"] != "<Non-Categorical Values>",
    ]
    if len(processed) > 0:
        num_values = processed[["Variable", "value"]].nunique()
        num_variables = processed["Variable"].nunique()
    else:
        num_values = 0
        num_variables = 0
    click.echo(
        click.style(
            f"Done: Saved {num_values:,} unique value counts for {num_variables:,} non-continuous variables to {output}",
            fg="green",
        )
    )


@describe_cli.command(help="Get the type of each variable")
@arg_data
@arg_output
def get_types(data, output):
    # Describe
    results = describe.get_types(data.df)
    # Save results
    results.to_csv(output, sep="\t", header=False)
    # Log
    click.echo(
        click.style(
            f"Done: Saved types of {len(results)} variables to {output}", fg="green"
        )
    )


@describe_cli.command(
    help="Report the percent of observations that are NA for each variable"
)
@arg_data
@arg_output
def percent_na(data, output):
    # Describe
    results = describe.percent_na(data.df)
    # Save results
    results.to_csv(output, sep="\t", index=False)
    # Log
    click.echo(
        click.style(
            f"Done: Saved results for {len(results):,} variables to {output}",
            fg="green",
        )
    )


@describe_cli.command(help="Report and test the skewness for each continuous variable")
@arg_data
@arg_output
@click.option(
    "--dropna/--keepna", default=False, help="Omit NA values before calculating skew"
)
def skewness(data, output, dropna):
    # Describe
    results = describe.skewness(data.df, dropna)
    # Save results
    results.to_csv(output, sep="\t", index=False)
    # Log
    click.echo(
        click.style(
            f"Done: Saved results for {len(results):,} variables to {output}",
            fg="green",
        )
    )
