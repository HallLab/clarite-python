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

from clarite.internal.regression import GLMRegression, WeightedGLMRegression
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
    # Covariates must be a list
    if type(covariates) != list:
        raise ValueError("'covariates' must be specified as a list.  Use an empty list ([]) if there aren't any.")

    # Make sure the index of each dataset is not a multiindex and give it a consistent name
    if isinstance(data.index, pd.MultiIndex):
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
    else:
        click.echo(f"Running {len(rv_bin):,} binary, {len(rv_cat):,} categorical, and {len(rv_cont):,} continuous variables")

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
    elif pheno_kind == 'constant':
        raise ValueError(f"The phenotype ('{phenotype}') is a constant value.")
    elif pheno_kind == 'categorical':
        raise NotImplementedError("Categorical Phenotypes are not yet supported.")
    elif pheno_kind == 'continuous':
        click.echo(f"Running EWAS on a Continuous Outcome (family = Gaussian)")
    elif pheno_kind == 'binary':
        # Set phenotype categories so that the higher number is a success
        categories = sorted(data[phenotype].unique(), reverse=True)
        cat_type = pd.api.types.CategoricalDtype(categories=categories, ordered=True)
        data[phenotype] = data[phenotype].astype(cat_type)
        click.echo(click.style(f"Running EWAS on a Binary Outcome (family = Binomial)\n"
                               f"\t(Success = '{categories[0]}', Failure = '{categories[1]}')", fg='green'))
    else:
        raise ValueError(f"The phenotype's type could not be determined.  Please report this error.")

    # Log Survey Design if it is being used
    if survey_design_spec is not None:
        click.echo(click.style(f"Using a Survey Design:\n{survey_design_spec}", fg='green'))

    # Run Regressions
    ewas_results = []
    rvs = rv_bin + rv_cat + rv_cont

    for rv in rvs:
        # Set up regression object
        if survey_design_spec is not None:
            regression = WeightedGLMRegression(
                data=data,
                outcome_variable=phenotype,
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
            warning_message = f"{rv} had warnings:"
            for warning in warnings:
                warning_message += f"\n\t{warning}"

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
