import click
import pandas as pd

from ...modules.survey import SurveyDesignSpec
from ...modules import analyze
from ..parameters import CLARITE_DATA, INPUT_FILE, OUTPUT_FILE


@click.group(name='analyze')
def analyze_cli():
    pass


@analyze_cli.command(help="Run an EWAS analysis")
@click.argument('phenotype', type=click.STRING)
@click.argument('bin-data', type=CLARITE_DATA)
@click.argument('cat-data', type=CLARITE_DATA)
@click.argument('cont-data', type=CLARITE_DATA)
@click.option('--covariate', '-c', multiple=True, help="Covariates")
@click.option('--covariance-calc', default='stata', type=click.Choice(['stata', 'jackknife']), help="Covariance calculation method")
@click.option('--min-n', default=200, type=click.IntRange(0, 999999), help="Minimum number of complete cases needed to run a regression")
@click.option('--survey-data', type=CLARITE_DATA, default=None,
              help="Tab-separated data file with survey weights, strata IDs, and/or cluster IDs.  Must have an 'ID' column.")
@click.option('--strata', type=click.STRING, default=None, help="Name of the strata column in the survey data")
@click.option('--cluster', type=click.STRING, default=None, help="Name of the cluster column in the survey data")
@click.option('--nested/--not-nested', help="Whether survey data is nested or not")
@click.option('--weights-file', type=INPUT_FILE, default=None,
              help="Tab-delimited data file with 'Variable' and 'weight' columns to match weights from the survey data to specific variables")
@click.option('--weight', '-w', type=click.STRING, default=None,
              help="Name of a survey weight column found in the survey data.  This option can't be used with --weights-file")
@click.option('--single-cluster', type=click.Choice(['error', 'scaled', 'centered', 'certainty']), default='error', help="How to handle singular clusters")
@click.argument('output', type=OUTPUT_FILE)
def ewas(phenotype, bin_data, cat_data, cont_data, covariate, covariance_calc, min_n,
         survey_data, strata, cluster, nested, weights_file, weight, single_cluster, output):
    """Run EWAS and add corrected pvalues"""
    # Make covariates into a list
    covariates = list(covariate)
    # Load optional survey data
    if survey_data is not None:
        if weights_file is not None and weight is not None:
            raise ValueError("Either 'weights-file' or 'weight' should be specified, not both.")
        elif weights_file is not None:
            weights = pd.read_csv(weights_file, sep="\t")
            if list(weights) != ['variable', 'weight']:
                raise ValueError(f"The weights-file must be a tab-separated file with two columns: 'variable' and 'weight'. "
                                 f"Columns were: {', '.join(list(weights))}")
            weights = weights.set_index('variable')['weight'].to_dict()
        elif weight is not None:
            weights = weight
        elif weights_file is None and weight is None:
            weights = None
        sd = SurveyDesignSpec(survey_data, strata=strata, cluster=cluster, nest=nested, weights=weights, single_cluster=single_cluster)
    else:
        sd = None
    # Remove variables with missing weights
    if type(weights) == dict:
        missing_weights_bin = set(list(bin_data)) - set([phenotype] + covariates) - set(weights.keys())
        for v in missing_weights_bin:
            click.echo(click.style(f"\tSkipping binary variable '{v}' because it wasn't listed in the weights file", fg='yellow'))
            bin_data = bin_data.drop(v, axis='columns')
        missing_weights_cat = set(list(cat_data)) - set([phenotype] + covariates) - set(weights.keys())
        for v in missing_weights_cat:
            click.echo(click.style(f"\tSkipping categorical variable '{v}' because it wasn't listed in the weights file", fg='yellow'))
            cat_data = cat_data.drop(v, axis='columns')
        missing_weights_cont = set(list(cont_data)) - set([phenotype] + covariates) - set(weights.keys())
        for v in missing_weights_cont:
            click.echo(click.style(f"\tSkipping continuous variable '{v}' because it wasn't listed in the weights file", fg='yellow'))
            cont_data = cont_data.drop(v, axis='columns')
    # Run ewas
    result = analyze.ewas(phenotype=phenotype, covariates=covariates, bin_df=bin_data, cat_df=cat_data, cont_df=cont_data,
                          survey_design_spec=sd, cov_method=covariance_calc, min_n=min_n)
    # Add corrected pvalues
    analyze.add_corrected_pvalues(result)
    # Save
    result.to_csv(output, sep="\t")
    # Log
    click.echo(click.style(f"Done: Saved EWAS results to {output}", fg='green'))

# TODO: Make this use an ewas result datatype
@analyze_cli.command(help="filter out non-significant results")
@click.argument('ewas_result_data', type=INPUT_FILE)
@click.argument('output', type=OUTPUT_FILE)
@click.option('--fdr/--bonferroni', 'use_fdr', default=True, help="Use FDR (--fdr) or Bonferroni pvalues (--bonferroni).  FDR by default.")
@click.option('--pvalue', '-p', type=click.FLOAT, default=0.05, help="Keep results with a pvalue <= this value (0.05 by default)")
def get_significant(ewas_result_data, output, use_fdr, pvalue):
    # Load data
    data = pd.read_csv(ewas_result_data, sep="\t", index_col=['variable', 'phenotype'])
    # Check columns
    if list(data) != analyze.result_columns + analyze.corrected_pvalue_columns:
        raise ValueError(f"{ewas_result_data} was not a valid EWAS result file.")
    # Filter
    if use_fdr:
        col = 'pvalue_fdr'
    else:
        col = 'pvalue_bonferroni'
    data = data.loc[data[col] <= pvalue, ]
    # Save result
    data.to_csv(output, sep="\t")
    # Log
    click.echo(click.style(f"Done: Saved {len(data):,} variables to {output}", fg='green'))
