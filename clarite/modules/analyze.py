"""
Analyze
========

EWAS and associated calculations

  .. autosummary::
     :toctree: modules/analyze

     ewas
     add_corrected_pvalues


"""
from pathlib import Path
from typing import List, Optional

import click
import pandas as pd
from numpy import nan
from statsmodels.stats.multitest import multipletests
from .survey import SurveyDesignSpec

from clarite.internal.regression import GLMRegression, WeightedGLMRegression
from ..internal.utilities import _get_dtypes, requires, validate_ewas_params

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
      * Binary variables are treated as continuous features, with values of 0 and 1.
      * The results of a likelihood ratio test are used for categorical variables, so no Beta values or SE are reported.
      * The regression family is automatically selected based on the type of the phenotype.
        * Continuous phenotypes use gaussian regression
        * Binary phenotypes use binomial regression (the larger of the two values is counted as "success")
      * Categorical variables run with a survey design will not report Diff_AIC

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
    # Copy data to avoid modifying the original, in case it is changed
    data = data.copy(deep=True)

    # Get lists of variables to regress and validate parameters
    rv_bin, rv_cat, rv_cont, pheno_kind = validate_ewas_params(covariates, data, phenotype, survey_design_spec)

    # Run Regressions
    ewas_results = []
    rvs = rv_bin + rv_cat + rv_cont

    for rv in rvs:
        # Set up regression object
        if survey_design_spec is not None:
            regression = WeightedGLMRegression(
                data=data,
                outcome_variable=phenotype,
                outcome_dtype=pheno_kind,
                test_variable=rv,
                covariates=covariates,
                min_n=min_n,
                survey_design_spec=survey_design_spec,
                cov_method=cov_method
            )
        else:
            regression = GLMRegression(
                data=data,
                outcome_variable=phenotype,
                outcome_dtype=pheno_kind,
                test_variable=rv,
                covariates=covariates,
                min_n=min_n
            )

        # Run
        result, warnings, error = regression.run()

        # Log errors and warnings
        if error is not None:
            click.echo(click.style(f"{rv} = NULL due to: {error}", fg='red'))
        if len(warnings) > 0:
            click.echo(click.style(f"{rv} had warnings:", fg='yellow'))
            for warning in warnings:
                click.echo(click.style(f"\t{warning}", fg='yellow'))

        # Collect result
        ewas_results.append(result)

    # Process Results
    ewas_result = pd.DataFrame(ewas_results)
    ewas_result['Phenotype'] = phenotype  # Add phenotype
    ewas_result = ewas_result.sort_values('pvalue').set_index(['Variable', 'Phenotype'])  # Sort and set index
    ewas_result = ewas_result[['Variable_type', 'Converged', 'N', 'Beta', 'SE', 'Variable_pvalue',
                               'LRT_pvalue', 'Diff_AIC', 'pvalue']]  # Sort columns
    click.echo("Completed EWAS\n")
    return ewas_result


@requires('rpy2')
def ewas_r(phenotype: str,
           covariates: List[str],
           data: pd.DataFrame,
           survey_design_spec: Optional[SurveyDesignSpec] = None,
           min_n: Optional[int] = 200):
    """
    Run EWAS using R
    """
    # Copy data to avoid modifying the original, in case it is changed
    data = data.copy(deep=True)

    # Get lists of variables to regress and validate parameters
    rv_bin, rv_cat, rv_cont, pheno_kind = validate_ewas_params(covariates, data, phenotype, survey_design_spec)

    # Make the first column "ID"
    data = data.reset_index(drop=False)
    data.columns = ["ID", ] + [c for c in data.columns if c != "ID"]

    # Source R script to define the function
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from ..r_code.r_utilities import ewasresult2py, df_pandas2r
    r_code_folder = (Path(__file__).parent.parent / 'r_code')
    filename = str(r_code_folder / "ewas_r.R")
    ro.r.source(filename)

    # Lists of variables and covariates
    dtypes = _get_dtypes(data)
    cat_vars = ro.StrVector(rv_bin + rv_cat)
    cont_vars = ro.StrVector(rv_cont)
    cat_covars = ro.StrVector([v for v in covariates if (dtypes.loc[v] == 'categorical') or (dtypes.loc[v] == 'binary')])
    cont_covars = ro.StrVector([v for v in covariates if dtypes.loc[v] == 'continuous'])

    # These lists must be passed as NULL if they are empty
    if len(cat_vars) == 0:
        cat_vars = ro.NULL
    if len(cont_vars) == 0:
        cont_vars = ro.NULL
    if len(cat_covars) == 0:
        cat_covars = ro.NULL
    if len(cont_covars) == 0:
        cont_covars = ro.NULL

    # Regression Family
    if pheno_kind == 'binary':
        regression_family = "binomial"
    elif pheno_kind == 'continuous':
        regression_family = 'gaussian'
    else:
        raise ValueError("Phenotype must be 'binary' or 'continuous'")

    # Run with or without survey design info
    if survey_design_spec is None:
        with ro.conversion.localconverter(ro.default_converter + pandas2ri.converter):
            data_r = df_pandas2r(data)
            result = ro.r.ewas(d=data_r, cat_vars=cat_vars, cont_vars=cont_vars, y=phenotype,
                               cat_covars=cat_covars, cont_covars=cont_covars,
                               regression_family=regression_family, min_n=min_n)
    else:
        # Merge weights into data
        data = pd.merge(data, survey_design_spec.weights, left_index=True, right_index=True, how='left')
        if survey_design_spec.single_weight:
            weights = survey_design_spec.weight_name
        elif survey_design_spec.multi_weight:
            weights = survey_design_spec.weight_names
        else:
            raise ValueError("Weights must be provided")
        # Gather optional parts of survey parameters
        kwargs = dict()
        # Cluster IDs
        if survey_design_spec.has_cluster:
            kwargs['ids'] = ro.Formula(f"~{survey_design_spec.cluster_name}")
            data[survey_design_spec.cluster_name] = survey_design_spec.cluster
        else:
            kwargs['ids'] = ro.Formula("~1")
        # Strata
        if survey_design_spec.has_strata:
            kwargs['strat'] = ro.Formula(f"~{survey_design_spec.strata_name}")
            data[survey_design_spec.strata_name] = survey_design_spec.strata
        # Nest
        if survey_design_spec.nest:
            kwargs['nest'] = True
        else:
            kwargs['nest'] = False
        # fpc
        if survey_design_spec.has_fpc:
            kwargs['fpc'] = ro.Formula(f"~{survey_design_spec.fpc_name}")
            data[survey_design_spec.fpc_name] = survey_design_spec.fpc
        # Single cluster setting
        ro.r(f'options("survey.lonely.psu"="{survey_design_spec.single_cluster}")')
        with ro.conversion.localconverter(ro.default_converter + pandas2ri.converter):
            data_r = df_pandas2r(data)
            result = ro.r.ewas(d=data_r, cat_vars=cat_vars, cont_vars=cont_vars, y=phenotype,
                               cat_covars=cat_covars, cont_covars=cont_covars,
                               regression_family=regression_family,
                               min_n=min_n,
                               weights=weights,
                               **kwargs)

    result = ewasresult2py(result)
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
