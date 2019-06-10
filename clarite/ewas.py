from typing import List, Optional

import numpy as np
import pandas as pd
import patsy
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from clarite.survey import SurveyDesignSpec, SurveyModel

from .utilities import make_bin, make_categorical, make_continuous, regTermTest


result_columns = ['variable_type', 'N', 'beta', 'SE', 'var_pvalue', 'LRT_pvalue', 'diff_AIC', 'pvalue']
corrected_pvalue_columns = ['pvalue_bonferroni', 'pvalue_fdr']


def ewas(
        phenotype: str,
        covariates: List[str],
        bin_df: Optional[pd.DataFrame],
        cat_df: Optional[pd.DataFrame],
        cont_df: Optional[pd.DataFrame],
        survey_design_spec: Optional[SurveyDesignSpec] = None,
        cov_method: Optional[str] = 'stata'):
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

    Returns
    -------
    df: pd.DataFrame
        EWAS results DataFrame with these columns: ['variable_type', 'N', 'beta', 'SE', 'var_pvalue', 'LRT_pvalue', 'diff_AIC', 'pvalue']

    Examples
    --------
    >>>ewas_discovery = clarite.ewas("logBMI", covariates, nhanes_discovery_bin, nhanes_discovery_cat, nhanes_discovery_cont)
    Running EWAS on a continuous variable
    """
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
        raise NotImplementedError("Binary Phenotypes are not yet supported")
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
    return run_regression(phenotype, covariates, df, rv_bin, rv_cat, rv_cont, pheno_kind, survey_design_spec, cov_method)


def check_covars(subset_df, covariates, regression_variable):
    """Return a list of covariates that have >1 value in the subset dataframe"""
    unique_values = subset_df[covariates].nunique()
    varying_covars = list(unique_values[unique_values > 1].index.values)
    non_varying_covars = list(unique_values[unique_values <= 1].index.values)

    if len(non_varying_covars) > 0:
        print(f"WARNING: {regression_variable} has non-varying covariates(s): {', '.join(non_varying_covars)}")
    return varying_covars


def run_regression(phenotype: str,
                   covariates: List[str],
                   df: pd.DataFrame,
                   rv_bin: List[str],
                   rv_cat: List[str],
                   rv_cont: List[str],
                   pheno_kind: str,
                   survey_design_spec: Optional[SurveyDesignSpec],
                   cov_method: Optional[str]):
    """Run a regressions on variables"""
    result = []

    # Must ensure phenotype is numerical for logistic regression
    if pheno_kind == "binary" or pheno_kind == "categorical":
        df[phenotype] = df[phenotype].cat.codes

    # Use the correct or specified family/link
    if pheno_kind == "continuous":
        family = sm.families.Gaussian(link=sm.families.links.identity)
    else:
        # TODO
        # Note: DoF might change
        raise NotImplementedError("Only continuous phenotypes are currently supported")

    # Continuous Variables
    print(f"\n####### Regressing {len(rv_cont)} Continuous Variables #######\n")
    for rv in rv_cont:
        # Create blank result
        var_result = {c: np.nan for c in result_columns}
        var_result['variable'] = rv
        var_result['variable_type'] = 'continuous'

        # Check covariates
        subset = df[~df[rv].isna()]
        if len(subset) == 0:
            print(f"{rv} = NULL due to: No non-null observations of {rv}")
            result.append(var_result)
            continue
        varying_covariates = check_covars(subset, covariates, rv)

        # Set up the formula
        x_vals = varying_covariates + [rv]
        formula = f"{phenotype} ~ " + " + ".join([f"C({var_name})" if str(df.dtypes[var_name]) == 'category' else var_name for var_name in x_vals])

        # Run the regression
        try:
            if survey_design_spec is None:
                # Regress
                est = smf.glm(formula, data=subset, family=family).fit(use_t=True)

                # Get Results
                N = est.nobs
                beta = est.params[rv]
                SE = est.bse[rv]
                var_pvalue = est.pvalues[rv]
            else:
                # Regress
                y, X = patsy.dmatrices(formula, subset, return_type='dataframe')
                survey_design, index = survey_design_spec.get_survey_design(rv, X.index)
                # Update y and X with the new index, which may be smaller due to missing weights
                y = y.loc[index]
                X = X.loc[index]
                model = SurveyModel(design=survey_design, model_class=sm.GLM, cov_method=cov_method,
                                    init_args=dict(family=family),
                                    fit_args=dict(use_t=True))
                model.fit(y=y, X=X)

                # Get Results
                rv_idx_list = [i for i, n in enumerate(X.columns) if rv in n]
                if len(rv_idx_list) != 1:
                    raise ValueError(f"Failed to find regression variable column in the results for {rv}")
                else:
                    rv_idx = rv_idx_list[0]
                N = X.shape[0]
                beta = model.params[rv_idx]
                SE = model.stderr[rv_idx]
                tval = np.abs(beta / SE)  # T statistic is the absolute value of beta / SE
                # Get degrees of freedom
                if model.design.has_clusters or model.design.has_strata:
                    dof = survey_design.get_dof(X)
                else:
                    dof = model.result.df_model
                var_pvalue = scipy.stats.t.sf(tval, df=dof)*2  # Two-sided t-test
        except Exception as e:
            print(f"{rv} = NULL due to: {e}")
            result.append(var_result)
            continue

        # Save and Return Results
        var_result['N'] = N
        var_result['beta'] = beta
        var_result['SE'] = SE
        var_result['var_pvalue'] = var_pvalue
        var_result['pvalue'] = var_pvalue
        result.append(var_result)

    print(f"\n####### Regressing {len(rv_bin)} Binary Variables #######\n")
    for rv in rv_bin:
        # Create blank result
        var_result = {c: np.nan for c in result_columns}
        var_result['variable'] = rv
        var_result['variable_type'] = 'binary'

        # Check covariates
        subset = df[~df[rv].isna()]
        if len(subset) == 0:
            print(f"{rv} = NULL due to: No non-null observations of {rv}")
            result.append(var_result)
            continue
        varying_covariates = check_covars(subset, covariates, rv)

        # Set up the formula
        x_vals = varying_covariates + [rv]
        formula = f"{phenotype} ~ " + " + ".join([f"C({var_name})" if str(df.dtypes[var_name]) == 'category' else var_name for var_name in x_vals])

        # Run the regression
        try:
            if survey_design_spec is None:
                # Regress
                est = smf.glm(formula, data=subset, family=family).fit(use_t=True)

                # Get Results
                # Categorical-type RVs get a different name in the results, and aren't always at the end (since categorical come before non-categorical)
                rv_keys = [k for k in est.params.keys() if rv in k]
                try:
                    assert len(rv_keys) == 1
                    rv_key = rv_keys[0]
                except AssertionError:
                    raise KeyError(f"Error extracting results for '{rv}', try renaming the variable")
                N = est.nobs
                beta = est.params[rv_key]
                SE = est.bse[rv_key]
                var_pvalue = est.pvalues[rv_key]
            else:
                # Regress
                y, X = patsy.dmatrices(formula, subset, return_type='dataframe')
                survey_design, index = survey_design_spec.get_survey_design(rv, X.index)
                # Update y and X with the new index, which may be smaller due to missing weights
                y = y.loc[index]
                X = X.loc[index]
                model = SurveyModel(design=survey_design, model_class=sm.GLM, cov_method=cov_method,
                                    init_args=dict(family=family),
                                    fit_args=dict(use_t=True))
                model.fit(y=y, X=X)

                # Get Results
                # Categorical-type RVs get a different name in the results,
                rv_idx_list = [i for i, n in enumerate(X.columns) if rv in n]
                if len(rv_idx_list) != 1:
                    raise ValueError(f"Failed to find regression variable column in the results for {rv}")
                else:
                    rv_idx = rv_idx_list[0]
                N = X.shape[0]
                beta = model.params[rv_idx]
                SE = model.stderr[rv_idx]
                tval = np.abs(beta / SE)  # T statistic is the absolute value of beta / SE
                # Get degrees of freedom
                if model.design.has_clusters or model.design.has_strata:
                    dof = survey_design.get_dof(X)
                else:
                    dof = model.result.df_model
                var_pvalue = scipy.stats.t.sf(tval, df=dof)*2  # Two-sided t-test
        except Exception as e:
            print(f"{rv} = NULL due to: {e}")
            result.append(var_result)
            continue

        # Save and Return Results
        var_result['N'] = N
        var_result['beta'] = beta
        var_result['SE'] = SE
        var_result['var_pvalue'] = var_pvalue
        var_result['pvalue'] = var_pvalue
        result.append(var_result)

    print(f"\n####### Regressing {len(rv_cat)} Categorical Variables #######\n")
    # The change in deviance between a model and a nested version (with n fewer predictors) follows a chi-square distribution with n DoF
    # See https://en.wikipedia.org/wiki/Deviance_(statistics)
    for rv in rv_cat:
        # Create blank result
        var_result = {c: np.nan for c in result_columns}
        var_result['variable'] = rv
        var_result['variable_type'] = 'categorical'

        # Check covariates
        # Note: Using a subset is required for categorical variables to ensure the restricted and full model use the same data
        subset = df[~df[rv].isna()]
        if len(subset) == 0:
            print(f"{rv} = NULL due to: No non-null observations of {rv}")
            result.append(var_result)
            continue
        varying_covariates = check_covars(subset, covariates, rv)

        # Run the regression and compare the results to the restricted regression model using only observations that include the rv

        # "Reduced" model that only includes covariates
        formula_restricted = f"{phenotype} ~ " + " + ".join([f"C({var_name})"
                                                             if str(df.dtypes[var_name]) == 'category'
                                                             else var_name for var_name in varying_covariates])
        try:
            if survey_design_spec is None:
                est_restricted = smf.glm(formula_restricted, data=subset, family=family).fit(use_t=True)
            else:
                y, X = patsy.dmatrices(formula_restricted, subset, return_type='dataframe')
                survey_design, index = survey_design_spec.get_survey_design(rv, X.index)
                # Update y and X with the new index, which may be smaller due to missing weights
                y = y.loc[index]
                X = X.loc[index]
                model_restricted = SurveyModel(design=survey_design, model_class=sm.GLM, cov_method=cov_method,
                                               init_args=dict(family=family),
                                               fit_args=dict(use_t=True))
                model_restricted.fit(y=y, X=X)
        except Exception as e:
            print(f"{rv} = NULL due to: {e}")
            result.append(var_result)
            continue

        # Full Model
        x_vals = varying_covariates + [rv]
        formula = f"{phenotype} ~ " + " + ".join([f"C({var_name})" if str(df.dtypes[var_name]) == 'category' else var_name for var_name in x_vals])
        try:
            if survey_design_spec is None:
                # Regress full model
                est = smf.glm(formula, data=subset, family=family).fit(use_t=True)
                # Calculate Results
                lrdf = (est_restricted.df_resid - est.df_resid)
                lrstat = -2*(est_restricted.llf - est.llf)
                lr_pvalue = scipy.stats.chi2.sf(lrstat, lrdf)
                # Gather Other Results
                N = est.nobs
                diff_AIC = est.aic - est_restricted.aic
            else:
                # Regress full model (Already have the survey_design object)
                y, X = patsy.dmatrices(formula, subset, return_type='dataframe')
                # Update y and X with the new index from the SurveyDesign, which may be smaller due to missing weights
                y = y.loc[index]
                X = X.loc[index]
                model = SurveyModel(design=survey_design, model_class=sm.GLM, cov_method=cov_method,
                                    init_args=dict(family=family),
                                    fit_args=dict(use_t=True))
                model.fit(y=y, X=X)
                # Calculate Results
                dof = survey_design.get_dof(X)
                N = X.shape[0]
                diff_AIC = model.result.aic - model_restricted.result.aic
                if model.design.has_strata or model.design.has_clusters:
                    # Calculate pvalue using vcov
                    lr_pvalue = regTermTest(full_model=model, restricted_model=model_restricted, ddf=dof, X_names=X.columns, var_name=rv)
                else:
                    # Calculate using llf from model results
                    lrdf = (model_restricted.result.df_resid - model.result.df_resid)
                    lrstat = -2*(model_restricted.result.llf - model.result.llf)
                    lr_pvalue = scipy.stats.chi2.sf(lrstat, lrdf)

        except Exception as e:
            print(f"{rv} = NULL due to: {e}")
            result.append(var_result)
            continue

        # Save and Return Results
        var_result['N'] = N
        var_result['LRT_pvalue'] = lr_pvalue
        var_result['diff_AIC'] = diff_AIC
        var_result['pvalue'] = lr_pvalue

        result.append(var_result)

    # Gather All Results
    result = pd.DataFrame(result)
    result['phenotype'] = phenotype  # Add phenotype
    result = result.sort_values('pvalue').set_index(['variable', 'phenotype'])  # Sort and set index
    result = result[['variable_type', 'N', 'beta', 'SE', 'var_pvalue', 'LRT_pvalue', 'diff_AIC', 'pvalue']]  # Sort columns
    return result


def add_corrected_pvalues(ewas_result):
    """
    Add bonferroni and FDR pvalues to an ewas result and sort by increasing FDR (in-place)

    Parameters
    ----------
    ewas_result: pd.DataFrame
        EWAS results DataFrame with these columns: ['variable_type', 'N', 'beta', 'SE', 'var_pvalue', 'LRT_pvalue', 'diff_AIC', 'pvalue']

    Returns
    -------
    None

    Examples
    --------
    >>>clarite.add_corrected_pvalues(ewas_discovery)
    """
    # TODO: These are slightly off from values in R- is it due to rounding?
    ewas_result['pvalue_bonferroni'] = multipletests(ewas_result['pvalue'].fillna(1.0), method="bonferroni")[1]
    ewas_result['pvalue_fdr'] = multipletests(ewas_result['pvalue'].fillna(1.0), method="fdr_bh")[1]
    ewas_result.sort_values(by=['pvalue_fdr', 'pvalue'], inplace=True)
