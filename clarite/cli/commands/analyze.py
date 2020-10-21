import click
import pandas as pd

from ...modules.survey import SurveyDesignSpec
from ...modules import analyze
from ..parameters import CLARITE_DATA, INPUT_FILE, EWAS_RESULT, arg_data, arg_output
from ..custom_types import save_clarite_ewas


@click.group(name="analyze")
def analyze_cli():
    pass


@analyze_cli.command(help="Run an EWAS analysis")
@click.argument("outcome", type=click.STRING)
@arg_data
@arg_output
@click.option("--covariate", "-c", multiple=True, help="Covariates")
@click.option(
    "--covariance-calc",
    default="stata",
    type=click.Choice(["stata", "jackknife"]),
    help="Covariance calculation method",
)
@click.option(
    "--min-n",
    default=200,
    type=click.IntRange(0, 999999),
    help="Minimum number of complete cases needed to run a regression",
)
@click.option(
    "--survey-data",
    type=CLARITE_DATA,
    default=None,
    help="Tab-separated data file with survey weights, strata IDs, and/or cluster IDs.  Must have an 'ID' column.",
)
@click.option(
    "--strata",
    type=click.STRING,
    default=None,
    help="Name of the strata column in the survey data",
)
@click.option(
    "--cluster",
    type=click.STRING,
    default=None,
    help="Name of the cluster column in the survey data",
)
@click.option("--nested/--not-nested", help="Whether survey data is nested or not")
@click.option(
    "--weights-file",
    type=INPUT_FILE,
    default=None,
    help="Tab-delimited data file with 'Variable' and 'Weight' columns to match weights from the survey data to specific variables",
)
@click.option(
    "--weight",
    "-w",
    type=click.STRING,
    default=None,
    help="Name of a survey weight column found in the survey data.  This option can't be used with --weights-file",
)
@click.option(
    "--fpc",
    type=click.STRING,
    default=None,
    help="Name of the finite population correction column in the survey data",
)
@click.option(
    "--single-cluster",
    type=click.Choice(["fail", "adjust", "average", "certainty"]),
    default="fail",
    help="How to handle singular clusters",
)
def ewas(
    outcome,
    data,
    output,
    covariate,
    covariance_calc,
    min_n,
    survey_data,
    strata,
    cluster,
    nested,
    weights_file,
    weight,
    fpc,
    single_cluster,
):
    """Run EWAS and add corrected pvalues"""
    # Keep just the data from the loaded CLARITE_DATA arguments/options
    data = data.df
    # Make covariates into a list
    covariates = list(covariate)
    # Load optional survey data
    if survey_data is not None:
        if weights_file is not None and weight is not None:
            raise ValueError(
                "Either 'weights-file' or 'weight' should be specified, not both."
            )
        elif weights_file is not None:
            weights = pd.read_csv(weights_file, sep="\t")
            if list(weights) != ["Variable", "Weight"]:
                raise ValueError(
                    f"The weights-file must be a tab-separated file with two columns: 'Variable' and 'Weight'. "
                    f"Columns were: {', '.join(list(weights))}"
                )
            weights = weights.set_index("Variable")["Weight"].to_dict()
        elif weight is not None:
            weights = weight
        elif weights_file is None and weight is None:
            weights = None
        sd = SurveyDesignSpec(
            survey_data.df,
            strata=strata,
            cluster=cluster,
            nest=nested,
            weights=weights,
            fpc=fpc,
            single_cluster=single_cluster,
        )
    else:
        sd = None
        weights = None
    # Remove variables with missing weights
    if type(weights) == dict:
        missing_weights = (
            set(list(data)) - set([outcome] + covariates) - set(weights.keys())
        )
        for v in missing_weights:
            click.echo(
                click.style(
                    f"\tWARNING: Skipping variable '{v}' because it wasn't listed in the weights file",
                    fg="yellow",
                )
            )
            data = data.drop(v, axis="columns")
    # Run ewas
    result = analyze.ewas(
        outcome=outcome,
        covariates=covariates,
        data=data,
        survey_design_spec=sd,
        cov_method=covariance_calc,
        min_n=min_n,
    )
    # Save
    save_clarite_ewas(result, output)


@analyze_cli.command(help="Run an EWAS analysis using R")
@click.argument("outcome", type=click.STRING)
@arg_data
@arg_output
@click.option("--covariate", "-c", multiple=True, help="Covariates")
@click.option(
    "--covariance-calc",
    default="stata",
    type=click.Choice(["stata", "jackknife"]),
    help="Covariance calculation method",
)
@click.option(
    "--min-n",
    default=200,
    type=click.IntRange(0, 999999),
    help="Minimum number of complete cases needed to run a regression",
)
@click.option(
    "--survey-data",
    type=CLARITE_DATA,
    default=None,
    help="Tab-separated data file with survey weights, strata IDs, and/or cluster IDs.  Must have an 'ID' column.",
)
@click.option(
    "--strata",
    type=click.STRING,
    default=None,
    help="Name of the strata column in the survey data",
)
@click.option(
    "--cluster",
    type=click.STRING,
    default=None,
    help="Name of the cluster column in the survey data",
)
@click.option("--nested/--not-nested", help="Whether survey data is nested or not")
@click.option(
    "--weights-file",
    type=INPUT_FILE,
    default=None,
    help="Tab-delimited data file with 'Variable' and 'Weight' columns to match weights from the survey data to specific variables",
)
@click.option(
    "--weight",
    "-w",
    type=click.STRING,
    default=None,
    help="Name of a survey weight column found in the survey data.  This option can't be used with --weights-file",
)
@click.option(
    "--fpc",
    type=click.STRING,
    default=None,
    help="Name of the finite population correction column in the survey data",
)
@click.option(
    "--single-cluster",
    type=click.Choice(["fail", "adjust", "average", "certainty"]),
    default="fail",
    help="How to handle singular clusters",
)
def ewas_r(
    outcome,
    data,
    output,
    covariate,
    covariance_calc,
    min_n,
    survey_data,
    strata,
    cluster,
    nested,
    weights_file,
    weight,
    fpc,
    single_cluster,
):
    """Run EWAS using R and add corrected pvalues"""
    # Keep just the data from the loaded CLARITE_DATA arguments/options
    data = data.df
    # Make covariates into a list
    covariates = list(covariate)
    # Load optional survey data
    if survey_data is not None:
        if weights_file is not None and weight is not None:
            raise ValueError(
                "Either 'weights-file' or 'weight' should be specified, not both."
            )
        elif weights_file is not None:
            weights = pd.read_csv(weights_file, sep="\t")
            if list(weights) != ["Variable", "Weight"]:
                raise ValueError(
                    f"The weights-file must be a tab-separated file with two columns: 'Variable' and 'Weight'. "
                    f"Columns were: {', '.join(list(weights))}"
                )
            weights = weights.set_index("Variable")["Weight"].to_dict()
        elif weight is not None:
            weights = weight
        elif weights_file is None and weight is None:
            weights = None
        sd = SurveyDesignSpec(
            survey_data.df,
            strata=strata,
            cluster=cluster,
            nest=nested,
            weights=weights,
            fpc=fpc,
            single_cluster=single_cluster,
        )
    else:
        sd = None
        weights = None
    # Remove variables with missing weights
    if type(weights) == dict:
        missing_weights = (
            set(list(data)) - set([outcome] + covariates) - set(weights.keys())
        )
        for v in missing_weights:
            click.echo(
                click.style(
                    f"\tWARNING: Skipping variable '{v}' because it wasn't listed in the weights file",
                    fg="yellow",
                )
            )
            data = data.drop(v, axis="columns")
    # Run ewas
    result = analyze.ewas_r(
        outcome=outcome,
        covariates=covariates,
        data=data,
        survey_design_spec=sd,
        cov_method=covariance_calc,
        min_n=min_n,
    )
    # Save
    save_clarite_ewas(result, output)


@analyze_cli.command(help="Get FDR-corrected and Bonferroni-corrected pvalues")
@click.argument("ewas_result", type=EWAS_RESULT)
@arg_output
def add_corrected_pvals(ewas_result, output):
    _, data = ewas_result
    analyze.add_corrected_pvalues(data)
    # Save result
    save_clarite_ewas(data, output)


@analyze_cli.command(help="filter out non-significant results")
@click.argument("ewas_result", type=EWAS_RESULT)
@arg_output
@click.option(
    "--fdr/--bonferroni",
    "use_fdr",
    default=True,
    help="Use FDR (--fdr) or Bonferroni pvalues (--bonferroni).  FDR by default.",
)
@click.option(
    "--pvalue",
    "-p",
    type=click.FLOAT,
    default=0.05,
    help="Keep results with a pvalue <= this value (0.05 by default)",
)
def get_significant(ewas_result, output, use_fdr, pvalue):
    # Filter
    if use_fdr:
        col = "pvalue_fdr"
    else:
        col = "pvalue_bonferroni"
    _, data = ewas_result
    data = data.loc[
        data[col] <= pvalue,
    ]
    # Save result
    save_clarite_ewas(data, output)
