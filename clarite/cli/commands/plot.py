import click
import pandas as pd
from matplotlib import pyplot as plt

from ...modules import plot
from ..parameters import arg_data, arg_output, EWAS_RESULT, INPUT_FILE


@click.group(name="plot")
def plot_cli():
    pass


@plot_cli.command(help="Create a histogram plot of a variable")
@arg_data
@arg_output
@click.argument("Variable", type=click.STRING)
def histogram(data, output, variable):
    # Plot
    plot.histogram(data=data.df, column=variable)
    # Save and Close
    plt.savefig(output)
    plt.close()
    # Log
    click.echo(click.style(f"Done: Saved plot to {output}", fg="green"))


@plot_cli.command(help="Generate a pdf containing distribution plots for each variable")
@arg_data
@arg_output
@click.option(
    "--kind",
    "-k",
    default="count",
    type=click.Choice(["count", "box", "violin", "qq"]),
    help="Kind of plot used for continuous data.  Non-continuous always shows a count plot.",
)
@click.option(
    "--nrows",
    default=4,
    type=click.IntRange(min=1, max=10),
    help="Number of rows per page",
)
@click.option(
    "--ncols",
    default=3,
    type=click.IntRange(min=1, max=10),
    help="Number of columns per page",
)
@click.option(
    "--quality",
    "-q",
    default="medium",
    type=click.Choice(["low", "medium", "high"]),
    help="Quality of the generated plots: low (150 dpi), medium (300 dpi), or high (1200 dpi).",
)
@click.option("--sort/--no-sort", help="Sort variables alphabetically")
def distributions(data, output, kind, nrows, ncols, quality, sort):
    # Plot and save
    plot.distributions(
        data=data.df,
        filename=output,
        continuous_kind=kind,
        nrows=nrows,
        ncols=ncols,
        quality=quality,
        sort=sort,
    )
    # Log
    click.echo(click.style(f"Done: Saved plot to {output}", fg="green"))


@plot_cli.command(help="Generate a manhattan plot of EWAS results")
@click.argument("ewas_result", type=EWAS_RESULT)
@arg_output
@click.option(
    "--categories",
    "-c",
    type=INPUT_FILE,
    default=None,
    help="tab-separate file with two columns: 'Variable' and 'category'",
)
@click.option(
    "--bonferroni",
    type=click.FLOAT,
    default=0.05,
    help="cutoff value to plot bonferroni-adjusted pvalue line",
)
@click.option(
    "--fdr",
    type=click.FLOAT,
    default=None,
    help="cutoff value to plot fdr-adjusted pvalue line",
)
@click.option(
    "--other",
    "-o",
    multiple=True,
    type=EWAS_RESULT,
    help="other datasets to include in the plot",
)
@click.option(
    "--nlabeled",
    default=3,
    type=click.IntRange(min=0, max=50),
    help="label top n points",
)
@click.option(
    "--label",
    default=None,
    multiple=True,
    type=click.STRING,
    help="label points by name",
)
def manhattan(ewas_result, output, categories, bonferroni, fdr, other, nlabeled, label):
    # Load data
    name, data = ewas_result
    data_dict = {name: data}
    for (name, data) in other:
        data_dict[name] = data
    # Load categories, if any
    if categories is not None:
        categories = pd.read_csv(categories, sep="\t")
        categories.columns = ["Variable", "category"]
        categories = categories.set_index("Variable")["category"].to_dict()
    # Plot and save
    plot.manhattan(
        data_dict,
        categories=categories,
        bonferroni=bonferroni,
        fdr=fdr,
        num_labeled=nlabeled,
        label_vars=label,
        filename=output,
    )
    # Log
    click.echo(click.style(f"Done: Saved plot to {output}", fg="green"))


@plot_cli.command(
    help="Generate a manhattan plot of EWAS results showing Bonferroni-corrected pvalues"
)
@click.argument("ewas_result", type=EWAS_RESULT)
@arg_output
@click.option(
    "--categories",
    "-c",
    type=INPUT_FILE,
    default=None,
    help="tab-separate file with two columns: 'Variable' and 'category'",
)
@click.option(
    "--cutoff",
    type=click.FLOAT,
    default=0.05,
    help="cutoff value for plotting the significance line",
)
@click.option(
    "--fdr",
    type=click.FLOAT,
    default=None,
    help="cutoff value to plot Bonferroni-adjusted pvalue line",
)
@click.option(
    "--other",
    "-o",
    multiple=True,
    type=EWAS_RESULT,
    help="other datasets to include in the plot",
)
@click.option(
    "--nlabeled",
    default=3,
    type=click.IntRange(min=0, max=50),
    help="label top n points",
)
@click.option(
    "--label",
    default=None,
    multiple=True,
    type=click.STRING,
    help="label points by name",
)
def manhattan_bonferroni(
    ewas_result, output, categories, cutoff, other, nlabeled, label
):
    # Load data
    name, data = ewas_result
    data_dict = {name: data}
    for (name, data) in other:
        data_dict[name] = data
    # Load categories, if any
    if categories is not None:
        categories = pd.read_csv(categories, sep="\t")
        categories.columns = ["Variable", "category"]
        categories = categories.set_index("Variable")["category"].to_dict()
    # Plot and save
    plot.manhattan_bonferroni(
        data_dict,
        categories=categories,
        cutoff=cutoff,
        num_labeled=nlabeled,
        label_vars=label,
        filename=output,
    )
    # Log
    click.echo(click.style(f"Done: Saved plot to {output}", fg="green"))


@plot_cli.command(
    help="Generate a manhattan plot of EWAS results showing FDR-corrected pvalues"
)
@click.argument("ewas_result", type=EWAS_RESULT)
@arg_output
@click.option(
    "--categories",
    "-c",
    type=INPUT_FILE,
    default=None,
    help="tab-separate file with two columns: 'Variable' and 'category'",
)
@click.option(
    "--cutoff",
    type=click.FLOAT,
    default=0.05,
    help="cutoff value for plotting the significance line",
)
@click.option(
    "--fdr",
    type=click.FLOAT,
    default=None,
    help="cutoff value to plot fdr-adjusted pvalue line",
)
@click.option(
    "--other",
    "-o",
    multiple=True,
    type=EWAS_RESULT,
    help="other datasets to include in the plot",
)
@click.option(
    "--nlabeled",
    default=3,
    type=click.IntRange(min=0, max=50),
    help="label top n points",
)
@click.option(
    "--label",
    default=None,
    multiple=True,
    type=click.STRING,
    help="label points by name",
)
def manhattan_fdr(ewas_result, output, categories, cutoff, other, nlabeled, label):
    # Load data
    name, data = ewas_result
    data_dict = {name: data}
    for (name, data) in other:
        data_dict[name] = data
    # Load categories, if any
    if categories is not None:
        categories = pd.read_csv(categories, sep="\t")
        categories.columns = ["Variable", "category"]
        categories = categories.set_index("Variable")["category"].to_dict()
    # Plot and save
    plot.manhattan_fdr(
        data_dict,
        categories=categories,
        cutoff=cutoff,
        num_labeled=nlabeled,
        label_vars=label,
        filename=output,
    )
    # Log
    click.echo(click.style(f"Done: Saved plot to {output}", fg="green"))


@click.argument("ewas_result", type=EWAS_RESULT)
@arg_output
@click.option(
    "--pvalue_name",
    type=click.Choice(["pvalue", "pvalue_bonferroni", "pvalue_FDR"]),
    default="pvalue",
    help="Which pvalues to use in the plot",
)
@click.option(
    "--cutoff",
    type=click.FLOAT,
    default=0.05,
    help="cutoff value for plotting the significance line",
)
@click.option(
    "--num_rows", type=click.INT, default=20, help="How many EWAS result rows to plot"
)
def top_results(ewas_result, output, pvalue_name, cutoff, num_rows):
    plot.top_results(
        ewas_result,
        pvalue_name=pvalue_name,
        cutoff=cutoff,
        num_rows=num_rows,
        filename=output,
    )
    # Log
    click.echo(click.style(f"Done: Saved plot to {output}", fg="green"))
