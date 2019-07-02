"""
Analyze
========

EWAS and associated calculations

  **DataFrame Accessor**: ``None``

  **CLI Command**: ``analyze``

  .. autosummary::
     :toctree: modules/analyze

     ewas
     add_corrected_pvalues


"""

from typing import List, Optional

import pandas as pd
from numpy import nan
from statsmodels.stats.multitest import multipletests
from .survey import SurveyDesignSpec

from .modify import make_binary, make_categorical, make_continuous, merge_variables
from ..internal.regression import Regression


result_columns = ['variable_type', 'converged', 'N', 'beta', 'SE', 'var_pvalue', 'LRT_pvalue', 'diff_AIC', 'pvalue']
corrected_pvalue_columns = ['pvalue_bonferroni', 'pvalue_fdr']


def ewas(
        phenotype: str,
        covariates: List[str],
        bin_df: Optional[pd.DataFrame],
        cat_df: Optional[pd.DataFrame],
        cont_df: Optional[pd.DataFrame],
        survey_design_spec: Optional[SurveyDesignSpec] = None,
        cov_method: Optional[str] = 'stata',
        min_n: Optional[int] = 200):
    """
    Run an EWAS on a phenotype

    Parameters
    ----------
    phenotype: string
        The variable to be used as the output of the regressions
    covariates: list (strings),
        The variables to be used as covariates.  Any variables in the DataFrames not listed as covariates are regressed.
    bin_df: pd.DataFrame or None
        A DataFrame containing binary variables
    cat_df: pd.DataFrame or None
        A DataFrame containing categorical variables
    cont_df: pd.DataFrame or None
        A DataFrame containing continuous variables
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
    >>> ewas_discovery = clarite.analyze.ewas("logBMI", covariates, nhanes_discovery_bin, nhanes_discovery_cat, nhanes_discovery_cont)
    Running EWAS on a continuous variable
    """
    # Merge the different datasets to ensure a consistent index, then reset it for simplicity in downstream processing
    dfs = [df for df in (bin_df, cat_df, cont_df) if df is not None]
    unified = dfs[0]
    for df in dfs[1:]:
        # Outer-merge on index
        unified = merge_variables(unified, df)
    # Reset index to a simple incrementing ID number for each observation
    unified = unified.reset_index(drop=True)
    unified.index.name = "ID"

    # Collects lists of regression variables and format dataframes using the new index from 'unified'
    rv_bin, rv_cat, rv_cont = list(), list(), list()
    if bin_df is not None:
        # Get original columns with new index and format as binary
        bin_df = make_binary(unified[list(bin_df)])
        rv_bin = [v for v in list(bin_df) if v not in covariates and v != phenotype]
    if cat_df is not None:
        # Get original columns with new index and format as categorical
        cat_df = make_categorical(unified[list(cat_df)])
        rv_cat = [v for v in list(cat_df) if v not in covariates and v != phenotype]
    if cont_df is not None:
        # Get original columns with new index and format as continuous
        cont_df = make_continuous(unified[list(cont_df)])
        rv_cont = [v for v in list(cont_df) if v not in covariates and v != phenotype]

    # Ensure covariates are all present
    variables = set(list(bin_df) + list(cat_df) + list(cont_df))
    missing_covariates = [c for c in covariates if c not in variables]
    if len(missing_covariates) > 0:
        raise ValueError(f"One or more covariates were not found in any of the input DataFrames: {', '.join(missing_covariates)}")

    # Ensure there are variables which can be regressed
    if len(rv_bin + rv_cat + rv_cont) == 0:
        raise ValueError(f"No variables are available to run regression on")

    # Figure out kind of phenotype
    if phenotype in list(bin_df):
        pheno_kind = 'binary'
    elif phenotype in list(cat_df):
        pheno_kind = 'categorical'
        raise NotImplementedError("Categorical Phenotypes are not yet supported")
    elif phenotype in list(cont_df):
        pheno_kind = 'continuous'
    else:
        raise ValueError(f"Couldn't find the phenotype ('{phenotype}') in the data.")
    print(f"Running EWAS on a {pheno_kind} variable")

    # Merge dfs if there are multiple
    dfs = [df for df in [bin_df, cat_df, cont_df] if df is not None]
    if len(dfs) == 1:
        df = dfs[0]
    else:
        df = dfs[0].join(dfs[1:], how="outer")

    # Run Regressions
    return run_regressions(phenotype, covariates, df, rv_bin, rv_cat, rv_cont, pheno_kind, min_n, survey_design_spec, cov_method)


def run_regressions(phenotype: str,
                    covariates: List[str],
                    df: pd.DataFrame,
                    rv_bin: List[str],
                    rv_cat: List[str],
                    rv_cont: List[str],
                    pheno_kind: str,
                    min_n: int,
                    survey_design_spec: Optional[SurveyDesignSpec],
                    cov_method: Optional[str]):
    """Run a regressions on variables"""
    result = []

    # Must ensure phenotype is numerical for logistic regression
    if pheno_kind == "binary" or pheno_kind == "categorical":
        df[phenotype] = df[phenotype].cat.codes

    # Continuous Variables
    print(f"\n####### Regressing {len(rv_cont)} Continuous Variables #######\n")
    for rv in rv_cont:
        # Set up the regression
        regression = Regression(variable=rv,
                                variable_kind='continuous',
                                phenotype=phenotype,
                                phenotype_kind=pheno_kind,
                                data=df,
                                covariates=covariates,
                                survey_design_spec=survey_design_spec,
                                cov_method=cov_method)
        # Run the regression
        try:
            regression.run(min_n=min_n)
        except Exception as e:
            print(f"{rv} = NULL due to: {e}")
        # Save results
        result.append(regression.get_results())

    print(f"\n####### Regressing {len(rv_bin)} Binary Variables #######\n")
    for rv in rv_bin:
        # Set up the regression
        regression = Regression(variable=rv,
                                variable_kind='binary',
                                phenotype=phenotype,
                                phenotype_kind=pheno_kind,
                                data=df,
                                covariates=covariates,
                                survey_design_spec=survey_design_spec,
                                cov_method=cov_method)
        # Run the regression
        try:
            regression.run(min_n=min_n)
        except Exception as e:
            print(f"{rv} = NULL due to: {e}")
        # Save results
        result.append(regression.get_results())

    print(f"\n####### Regressing {len(rv_cat)} Categorical Variables #######\n")
    for rv in rv_cat:
        # Set up the regression
        regression = Regression(variable=rv,
                                variable_kind='categorical',
                                phenotype=phenotype,
                                phenotype_kind=pheno_kind,
                                data=df,
                                covariates=covariates,
                                survey_design_spec=survey_design_spec,
                                cov_method=cov_method)
        # Run the regression
        try:
            regression.run(min_n=min_n)
        except Exception as e:
            print(f"{rv} = NULL due to: {e}")
        # Save results
        result.append(regression.get_results())

    # Gather All Results
    result = pd.DataFrame(result)
    result['phenotype'] = phenotype  # Add phenotype
    result = result.sort_values('pvalue').set_index(['variable', 'phenotype'])  # Sort and set index
    result = result[['variable_type', 'converged', 'N', 'beta', 'SE', 'var_pvalue', 'LRT_pvalue', 'diff_AIC', 'pvalue']]  # Sort columns
    return result


def add_corrected_pvalues(ewas_result):
    """
    Add bonferroni and FDR pvalues to an ewas result and sort by increasing FDR (in-place)

    Parameters
    ----------
    ewas_result: pd.DataFrame
        EWAS results DataFrame with these columns: ['variable_type', 'converged', N', 'beta', 'SE', 'var_pvalue', 'LRT_pvalue', 'diff_AIC', 'pvalue']

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
        ewas_result.sort_values(by=['pvalue_fdr', 'pvalue', 'converged'], inplace=True)
