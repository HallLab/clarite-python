"""
Analyze
========

EWAS and associated calculations

  .. autosummary::
     :toctree: modules/analyze

     ewas
     add_corrected_pvalues


"""

from typing import List, Optional

import click
import pandas as pd
from numpy import nan
from statsmodels.stats.multitest import multipletests
from .survey import SurveyDesignSpec

from ..internal.regression import Regression
from ..internal.utilities import _get_dtypes


result_columns = ['Variable_type', 'Converged', 'N', 'Beta', 'SE', 'Variable_pvalue',
                  'LRT_pvalue', 'Diff_AIC', 'pvalue']
corrected_pvalue_columns = ['pvalue_bonferroni', 'pvalue_fdr']


def ewas(
        phenotype: str,
        covariates: List[str],
        data: pd.DataFrame,
        survey_design_spec: Optional[SurveyDesignSpec] = None,
        cov_method: Optional[str] = 'stata',
        min_n: Optional[int] = 200):
    """
    Run an EWAS on a phenotype.

    Note:
      * Binary variables are treated as continuous, with values of 0 and 1.
      * The results of a likelihood ratio test are used for categorical variables, so no Beta values or SE are reported.

    Parameters
    ----------
    phenotype: string
        The variable to be used as the output of the regressions
    covariates: list (strings),
        The variables to be used as covariates.  Any variables in the DataFrames not listed as covariates are regressed.
    data: pd.DataFrame
        The data to be analyzed, including the phenotype, covariates, and any variables to be regressed.
    survey_design_spec: SurveyDesignSpec or None
        A SurveyDesignSpec object is used to create SurveyDesign objects for each regression.
    cov_method: str or None
        Covariance calculation method (if survey_design_spec is passed in).  'stata' or 'jackknife'
    min_n: int or None
        Minimum number of complete-case observations (no NA values for phenotype, covariates, variable, or weight)
        Defaults to 200

    Returns
    -------
    df: pd.DataFrame
        EWAS results DataFrame with these columns: ['variable_type', 'N', 'beta', 'SE', 'var_pvalue', 'LRT_pvalue', 'diff_AIC', 'pvalue']

    Examples
    --------
    >>> ewas_discovery = clarite.analyze.ewas("logBMI", covariates, nhanes_discovery)
    Running EWAS on a continuous variable
    """
    # Covariates must be a list
    if type(covariates) != list:
        raise ValueError("'covariates' must be specified as a list.  Use an empty list ([]) if there aren't any.")

    # Make sure the index of each dataset is not a multiindex and give it a consistent name
    if isinstance(data.index, pd.core.index.MultiIndex):
        raise ValueError(f"Data must not have a multiindex")
    data.index.name = "ID"

    # Collects lists of regression variables
    types = _get_dtypes(data)

    rv_bin = [v for v, t in types.iteritems() if t == 'binary' and v not in covariates and v != phenotype]
    rv_cat = [v for v, t in types.iteritems() if t == 'categorical' and v not in covariates and v != phenotype]
    rv_cont = [v for v, t in types.iteritems() if t == 'continuous' and v not in covariates and v != phenotype]

    # Ensure there are variables which can be regressed
    if len(rv_bin + rv_cat + rv_cont) == 0:
        raise ValueError(f"No variables are available to run regression on")

    # Ensure covariates are all present and not unknown type
    covariate_types = [types.get(c, None) for c in covariates]
    missing_covariates = [c for c, dt in zip(covariates, covariate_types) if dt is None]
    unknown_covariates = [c for c, dt in zip(covariates, covariate_types) if dt == 'unknown']
    if len(missing_covariates) > 0:
        raise ValueError(f"One or more covariates were not found in the data: {', '.join(missing_covariates)}")
    if len(unknown_covariates) > 0:
        raise ValueError(f"One or more covariates have an unknown datatype: {', '.join(unknown_covariates)}")

    # Validate the type of the phenotype variable
    pheno_kind = types.get(phenotype, None)
    if phenotype in covariates:
        raise ValueError(f"The phenotype ('{phenotype}') cannot also be a covariate.")
    elif pheno_kind is None:
        raise ValueError(f"The phenotype ('{phenotype}') was not found in the data.")
    elif pheno_kind == 'unknown':
        raise ValueError(f"The phenotype ('{phenotype}') has an unknown type.")
    elif pheno_kind == 'categorical':
        raise NotImplementedError("Categorical Phenotypes are not yet supported.")
    else:
        click.echo(f"Running EWAS on a {pheno_kind} variable")

    # Run Regressions
    return run_regressions(phenotype, covariates, data, rv_bin, rv_cat, rv_cont, pheno_kind, min_n, survey_design_spec, cov_method)


def run_regressions(phenotype: str,
                    covariates: List[str],
                    data: pd.DataFrame,
                    rv_bin: List[str],
                    rv_cat: List[str],
                    rv_cont: List[str],
                    pheno_kind: str,
                    min_n: int,
                    survey_design_spec: Optional[SurveyDesignSpec],
                    cov_method: Optional[str]):
    """Run a regressions on variables"""
    result = []

    # Continuous Variables
    click.echo(f"\n####### Regressing {len(rv_cont)} Continuous Variables #######\n")
    for rv in rv_cont:
        # Set up the regression
        regression = Regression(variable=rv,
                                variable_kind='continuous',
                                phenotype=phenotype,
                                phenotype_kind=pheno_kind,
                                data=data,
                                covariates=covariates,
                                survey_design_spec=survey_design_spec,
                                cov_method=cov_method)
        # Run the regression
        try:
            regression.run(min_n=min_n)
        except Exception as e:
            click.echo(f"{rv} = NULL due to: {e}")
        # Save results
        result.append(regression.get_results())

    click.echo(f"\n####### Regressing {len(rv_bin)} Binary Variables #######\n")
    for rv in rv_bin:
        # Set up the regression
        regression = Regression(variable=rv,
                                variable_kind='binary',
                                phenotype=phenotype,
                                phenotype_kind=pheno_kind,
                                data=data,
                                covariates=covariates,
                                survey_design_spec=survey_design_spec,
                                cov_method=cov_method)
        # Run the regression
        try:
            regression.run(min_n=min_n)
        except Exception as e:
            click.echo(f"{rv} = NULL due to: {e}")
        # Save results
        result.append(regression.get_results())

    click.echo(f"\n####### Regressing {len(rv_cat)} Categorical Variables #######\n")
    for rv in rv_cat:
        # Set up the regression
        regression = Regression(variable=rv,
                                variable_kind='categorical',
                                phenotype=phenotype,
                                phenotype_kind=pheno_kind,
                                data=data,
                                covariates=covariates,
                                survey_design_spec=survey_design_spec,
                                cov_method=cov_method)
        # Run the regression
        try:
            regression.run(min_n=min_n)
        except Exception as e:
            click.echo(f"{rv} = NULL due to: {e}")
        # Save results
        result.append(regression.get_results())

    # Gather All Results
    result = pd.DataFrame(result)
    result['Phenotype'] = phenotype  # Add phenotype
    result = result.sort_values('pvalue').set_index(['Variable', 'Phenotype'])  # Sort and set index
    result = result[['Variable_type', 'Converged', 'N', 'Beta', 'SE', 'Variable_pvalue',
                     'LRT_pvalue', 'Diff_AIC', 'pvalue']]  # Sort columns
    click.echo("Completed EWAS\n")
    return result


def add_corrected_pvalues(ewas_result):
    """
    Add bonferroni and FDR pvalues to an ewas result and sort by increasing FDR (in-place)

    Parameters
    ----------
    ewas_result: pd.DataFrame
        EWAS results DataFrame with these columns: ['Variable_type', 'Converged', 'N', 'Beta', 'SE', 'Variable_pvalue', 'LRT_pvalue', 'Diff_AIC', 'pvalue']

    Returns
    -------
    None

    Examples
    --------
    >>> clarite.analyze.add_corrected_pvalues(ewas_discovery)
    """
    # NA by default
    ewas_result['pvalue_bonferroni'] = nan
    ewas_result['pvalue_fdr'] = nan
    if (~ewas_result['pvalue'].isna()).sum() > 0:
        # Calculate values, ignoring NA pvalues
        ewas_result.loc[~ewas_result['pvalue'].isna(), 'pvalue_bonferroni'] = multipletests(ewas_result.loc[~ewas_result['pvalue'].isna(), 'pvalue'],
                                                                                            method="bonferroni")[1]
        ewas_result.loc[~ewas_result['pvalue'].isna(), 'pvalue_fdr'] = multipletests(ewas_result.loc[~ewas_result['pvalue'].isna(), 'pvalue'],
                                                                                     method="fdr_bh")[1]
        ewas_result.sort_values(by=['pvalue_fdr', 'pvalue', 'Converged'], inplace=True)
