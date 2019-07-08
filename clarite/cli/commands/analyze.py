import click
import pandas as pd

from ...modules.survey import SurveyDesignSpec
from ...modules import analyze, io
from ..parameters import input_file, output_file


@click.group(name='analyze')
def analyze_cli():
    pass


@analyze_cli.command(help="Run an EWAS analysis")
@click.argument('phenotype', type=click.STRING)
@click.argument('bin-data', type=input_file)
@click.argument('cat-data', type=input_file)
@click.argument('cont-data', type=input_file)
@click.option('--covariate', '-c', multiple=True, help="Covariates")
@click.option('--covariance-calc', default='stata', type=click.Choice(['stata', 'jackknife']), help="Covariance calculation method")
@click.option('--min-n', default=200, type=click.IntRange(0, 999999), help="Minimum number of complete cases needed to run a regression")
@click.option('--survey-data', type=input_file, default=None,
              help="Tab-separated data file with survey weights, strata IDs, and/or cluster IDs.  Must have an 'ID' column.")
@click.option('--strata', type=click.STRING, default=None, help="Name of the strata column in the survey data")
@click.option('--cluster', type=click.STRING, default=None, help="Name of the cluster column in the survey data")
@click.option('--nested/--not-nested', help="Whether survey data is nested or not")
@click.option('--weights-file', '-w', type=input_file, default=None,
              help="Tab-delimited data file with 'Variable' and 'weight' columns to match weights from the survey data to specific variables")
@click.option('--weight', '-w', type=click.STRING, default=None,
              help="Name of a survey weight column found in the survey data.  This option can't be used with --weights-file")
@click.option('--single-cluster', type=click.Choice(['error', 'scaled', 'centered', 'certainty']), default='error', help="How to handle singular clusters")
@click.argument('output', type=output_file)
def ewas(phenotype, bin_data, cat_data, cont_data, covariate, covariance_calc, min_n,
         survey_data, strata, cluster, nested, weights_file, weight, single_cluster, output):
    """Run EWAS and add corrected pvalues"""
    # Load data
    bin_data = io.load_data(bin_data)
    cat_data = io.load_data(cat_data)
    cont_data = io.load_data(cont_data)
    # Make covariates into a list
    covariates = list(covariate)
    # Load optional survey data
    if survey_data is not None:
        survey_data = pd.read_csv(survey_data, sep="\t")
        if weights_file is not None and weight is not None:
            raise ValueError("Either 'weights-file' or 'weight' should be specified, not both.")
        elif weights_file is not None:
            weights = pd.read_csv(weights_file, sep="\t")
        elif weight is not None:
            weights = weight
        elif weights_file is None and weight is None:
            weights = None
        sd = SurveyDesignSpec(survey_data, strata=strata, cluster=cluster, nest=nested, weights=weights, single_cluster=single_cluster)
    else:
        sd = None
    # Run ewas
    result = analyze.ewas(phenotype=phenotype, covariates=covariates, bin_df=bin_data, cat_df=cat_data, cont_df=cont_data,
                          survey_design_spec=sd, cov_method=covariance_calc, min_n=min_n)
    # Add corrected pvalues
    analyze.add_corrected_pvalues(result)
    # Save
    result.to_csv(output, sep="\t")
    # Log
    click.echo(click.style(f"Done: Saved EWAS results to {output}", fg='green'))
