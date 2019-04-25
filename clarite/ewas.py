from typing import List, Optional

import numpy as np
import pandas as pd
import scipy
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from .utilities import make_bin, make_categorical, make_continuous


result_columns = ['variable_type', 'N', 'beta', 'SE', 'var_pvalue', 'LRT_pvalue', 'diff_AIC', 'pvalue']
corrected_pvalue_columns = ['pvalue_bonferroni', 'pvalue_fdr']


def ewas(phenotype: str,
         covariates: List[str],
         bin_df: Optional[pd.DataFrame],
         cat_df: Optional[pd.DataFrame],
         cont_df: Optional[pd.DataFrame],
         groups: Optional[pd.Series],
         weights: Optional[pd.DataFrame]):
    """Run an EWAS"""
    # Process variable inputs
    rv_bin, rv_cat, rv_cont = list(), list(), list()
    if bin_df is not None:
        bin_df = make_bin(bin_df)
        rv_bin = [v for v in list(bin_df) if v not in covariates and v != phenotype]
    if cat_df is not None:
        cat_df = make_categorical(cat_df)
        rv_cat = [v for v in list(cat_df) if v not in covariates and v != phenotype]
    if cont_df is not None:
        cont_df = make_continuous(cont_df)
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

    # Return Regression Results
    return run_regression(phenotype, covariates, df, rv_bin, rv_cat, rv_cont, pheno_kind, groups, weights)


def clean_covars(df, covariates, phenotype, regression_variable):
    """Return a subset of the dataframe where regression_variable is not NA.  Also return a list of covariates that have >1 value in this subset"""
    unique_values = df.loc[~df[regression_variable].isna(), covariates].nunique()
    varying_covars = list(unique_values[unique_values > 1].index.values)
    non_varying_covars = list(unique_values[unique_values <= 1].index.values)

    if len(non_varying_covars) > 0:
        print(f"WARNING: {regression_variable} has non-varying covariates(s) after removing NA observations: {', '.join(non_varying_covars)}")
    return df.loc[~df[regression_variable].isna(), varying_covars + [phenotype, regression_variable]], varying_covars


def run_regression(phenotype: str,
                   covariates: List[str],
                   df: pd.DataFrame,
                   rv_bin: List[str],
                   rv_cat: List[str],
                   rv_cont: List[str],
                   pheno_kind: str,
                   groups: Optional[pd.Series],
                   weights: Optional[pd.DataFrame]):
    """Run a regression on continuous variables"""
    result = []

    # Must ensure phenotype is numerical for logistic regression
    if pheno_kind == "binary" or pheno_kind == "categorical":
        df[phenotype] = df[phenotype].cat.codes
        regression_kind = "logit"
    else:
        regression_kind = "ols"

    for rv in rv_cont:
        # Create blank result
        var_result = {c: np.nan for c in result_columns}
        var_result['variable'] = rv
        var_result['variable_type'] = 'continuous'

        # Check covariates
        subset, varying_covariates = clean_covars(df, covariates, phenotype, rv)

        # Run the regression and get results directly
        x_vals = varying_covariates + [rv]
        formula = f"{phenotype} ~ " + " + ".join([f"C({var_name})" if str(df.dtypes[var_name]) == 'category' else var_name for var_name in x_vals])
        try:
            if regression_kind == "ols":
                est = smf.ols(formula, data=subset).fit()
            elif regression_kind == 'logit':
                est = smf.logit(formula, data=subset).fit(disp=False)
        except Exception as e:
            print(f"{rv} = NULL due to: {e}")
            result.append(var_result)
            continue

        var_result['N'] = est.nobs
        var_result['beta'] = est.params[rv]
        var_result['SE'] = est.bse[rv]
        var_result['var_pvalue'] = est.pvalues[rv]
        var_result['pvalue'] = est.pvalues[rv]
        result.append(var_result)

    for rv in rv_bin:
        # Create blank result
        var_result = {c: np.nan for c in result_columns}
        var_result['variable'] = rv
        var_result['variable_type'] = 'binary'

        # Check covariates
        subset, varying_covariates = clean_covars(df, covariates, phenotype, rv)

        # Run the regression, and get results for the one kept variable value
        x_vals = varying_covariates + [rv]
        formula = f"{phenotype} ~ " + " + ".join([f"C({var_name})" if str(df.dtypes[var_name]) == 'category' else var_name for var_name in x_vals])
        try:
            if regression_kind == "ols":
                est = smf.ols(formula, data=subset, missing='drop').fit()
            elif regression_kind == 'logit':
                est = smf.logit(formula, data=subset, missing='drop').fit(disp=False)
        except Exception as e:
            print(f"{rv} = NULL due to: {e}")
            result.append(var_result)
            continue

        # Categorical RVs get a different name in the results, and aren't always at the end (since categorical come before non-categorical)
        rv_keys = [k for k in est.params.keys() if rv in k]
        try:
            assert len(rv_keys) == 1
            rv_key = rv_keys[0]
        except AssertionError:
            raise KeyError(f"Error extracting results for '{rv}', try renaming the variable")

        var_result['N'] = est.nobs
        var_result['beta'] = est.params[rv_key]
        var_result['SE'] = est.bse[rv_key]
        var_result['var_pvalue'] = est.pvalues[rv_key]
        var_result['pvalue'] = est.pvalues[rv_key]
        result.append(var_result)

    for rv in rv_cat:
        # Create blank result
        var_result = {c: np.nan for c in result_columns}
        var_result['variable'] = rv
        var_result['variable_type'] = 'categorical'

        # Check covariates
        subset, varying_covariates = clean_covars(df, covariates, phenotype, rv)

        # Run the regression and compare the results to the restricted regression model using only observations that include the rv

        # "Reduced" model that only includes covariates
        formula_restricted = f"{phenotype} ~ " + " + ".join([
            f"C({var_name})"
            if str(df.dtypes[var_name]) == 'category'
            else var_name for var_name in varying_covariates])
        try:
            if regression_kind == "ols":
                est_restricted = smf.ols(formula_restricted, data=subset).fit()
            elif regression_kind == 'logit':
                est_restricted = smf.logit(formula_restricted, data=subset).fit(disp=False)
        except Exception as e:
            print(f"{rv} = NULL due to error in runing the restricted model: {e}")
            result.append(var_result)
            continue

        x_vals = varying_covariates + [rv]
        formula = f"{phenotype} ~ " + " + ".join([f"C({var_name})" if str(df.dtypes[var_name]) == 'category' else var_name for var_name in x_vals])

        try:
            if regression_kind == "ols":
                est = smf.ols(formula, data=subset).fit()
            elif regression_kind == 'logit':
                est = smf.logit(formula, data=subset).fit(disp=False)
        except Exception as e:
            print(f"{rv} = NULL due to: {e}")
            result.append(var_result)
            continue

        lrdf = (est_restricted.df_resid - est.df_resid)
        lrstat = -2*(est_restricted.llf - est.llf)
        lr_pvalue = scipy.stats.chi2.sf(lrstat, lrdf)

        var_result['N'] = est.nobs
        var_result['LRT_pvalue'] = lr_pvalue
        var_result['diff_AIC'] = est.aic - est_restricted.aic
        var_result['pvalue'] = lr_pvalue
        result.append(var_result)

    # Gather Results
    result = pd.DataFrame(result)
    result['phenotype'] = phenotype  # Add phenotype
    result = result.sort_values('pvalue').set_index(['variable', 'phenotype'])  # Sort and set index
    result = result[['variable_type', 'N', 'beta', 'SE', 'var_pvalue', 'LRT_pvalue', 'diff_AIC', 'pvalue']]  # Sort columns
    return result


def add_corrected_pvalues(ewas_result):
    """Add bonferroni and FDR pvalues to an ewas result"""
    ewas_result['pvalue_bonferroni'] = multipletests(ewas_result['pvalue'], method="bonferroni", is_sorted=True)[1]
    ewas_result['pvalue_fdr'] = multipletests(ewas_result['pvalue'], method="fdr_bh", is_sorted=True)[1]
