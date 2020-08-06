from typing import Dict, Optional, List, Tuple

import click
import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf

from clarite.internal.utilities import _remove_empty_categories, _get_dtype

from .base import Regression


class GLMRegression(Regression):
    """
    Statsmodels GLM Regression.

    Note:
      * Binary variables are treated as continuous features, with values of 0 and 1.
      * The results of a likelihood ratio test are used for categorical variables, so no Beta values or SE are reported.
      * The regression family is automatically selected based on the type of the phenotype.
        * Continuous phenotypes use gaussian regression
        * Binary phenotypes use binomial regression (the larger of the two values is counted as "success")
      * Categorical variables run with a survey design will not report Diff_AIC

    Parameters
    ----------
    data: pd.DataFrame
        The data to be analyzed, including the phenotype, covariates, and any variables to be regressed.
    outcome_variable: string
        The variable to be used as the output (y) of the regression
    covariates: list (strings),
        The variables to be used as covariates.  Any variables in the DataFrames not listed as covariates are regressed.
    min_n: int or None
        Minimum number of complete-case observations (no NA values for phenotype, covariates, or variable)
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
    def __init__(self, data, outcome_variable, covariates, min_n=200):
        """
        Parameters
        ----------
        data - pd.DataFrame
        outcome_variable - name of the outcome variable
        covariates - other variables to include in the regression formula

        Kwargs
        ______
        min_n - minimum number of observations (after discarding any with NA)
        """
        # base class init
        # This takes in minimal regression params (data, outcome_variable, covariates) and
        # initializes additional parameters (outcome dtype, regression variables, error, and warnings)
        super().__init__(data=data,
                         outcome_variable=outcome_variable,
                         covariates=covariates)

        # Custom init involving kwargs passed to this regression
        self.min_n = min_n

        # Placeholders for some strings used to store information that is relayed when printing the object
        self.outcome_dtype_str = ""
        self.outcome_missing_str = ""

        # Ensure the data output type is compatible
        # Set 'self.family' and 'self.use_t' which are dependent on the outcome dtype
        # outcome_dtype_str is used during self.__str__
        if self.outcome_dtype == 'categorical':
            raise NotImplementedError("Categorical Phenotypes are not yet supported for this type of regression.")
        elif self.outcome_dtype == 'continuous':
            self.outcome_dtype_str = f"Continuous Outcome (family = Gaussian): '{self.outcome_variable}'"
            self.family = sm.families.Gaussian(link=sm.families.links.identity())
            self.use_t = True
        elif self.outcome_dtype == 'binary':
            # Use the order according to the categorical
            counts = self.data[self.outcome_variable].value_counts().to_dict()
            categories = self.data[self.outcome_variable].cat.categories
            codes, categories = zip(*enumerate(categories))
            self.data[self.outcome_variable].replace(categories, codes, inplace=True)
            self.outcome_dtype_str = f"Binary Outcome (family = Binomial): '{self.outcome_variable}'\n" \
                                     f"\t{counts[categories[0]]:,} occurrences of '{categories[0]}' coded as 0\n" \
                                     f"\t{counts[categories[1]]:,} occurrences of '{categories[1]}' coded as 1"
            self.family = sm.families.Binomial(link=sm.families.links.logit())
            self.use_t = False
        else:
            raise ValueError("The outcome variable's type could not be determined.  Please report this error.")

        # Log missing outcome values
        na_outcome_count = self.data[self.outcome_variable].isna().sum()
        self.outcome_missing_str = f"Using {len(self.data) - na_outcome_count:,} of {len(self.data):,} observations"
        if na_outcome_count > 0:
            self.outcome_missing_str += f"\n\t{na_outcome_count:,} are missing a value for the outcome variable"

    def __str__(self):
        string = f"{self.__class__.__name__}\n" \
                 f"{self.outcome_dtype_str}\n" \
                 f"{self.outcome_missing_str}\n" \
                 f"Regressing {len(self.results):,} variables\n"
        for k, v in self.regression_variables.items():
            string += f"\t{len(v):,} {k} variables\n"
        return string

    @staticmethod
    def get_default_result_dict():
        return {'Converged': False,
                'N': np.nan,
                'Beta': np.nan,
                'SE': np.nan,
                'Variable_pvalue': np.nan,
                'LRT_pvalue': np.nan,
                'Diff_AIC': np.nan,
                'pvalue': np.nan}

    def get_complete_case_idx(self, data, regression_variable):
        """Get index of observations that are not missing in the test variable, the outcome variable, and covariates"""
        return data.dropna(axis='index', how='any',
                           subset=[regression_variable, self.outcome_variable] + self.covariates)\
                   .index

    def check_covariate_values(self, complete_case_idx) -> Tuple[List[str], List[str]]:
        """Remove covariates that do not vary, warning when this occurs"""
        warnings = []
        unique_values = self.data.loc[complete_case_idx, self.covariates].nunique()
        varying_covars = list(unique_values[unique_values > 1].index.values)
        non_varying_covars = list(unique_values[unique_values <= 1].index.values)
        if len(non_varying_covars) > 0:
            warnings.append(f"non-varying covariates(s): {', '.join(non_varying_covars)}")
        return varying_covars, warnings

    def get_formulas(self, regression_variable, varying_covars) -> Tuple[str, str]:
        # Restricted Formula, just outcome and covariates
        formula_restricted = f"{self.outcome_variable} ~ "
        formula_restricted += " + ".join(
            [f"C({var_name})"
             if str(self.data.dtypes[var_name]) == 'category'
             else var_name for var_name in varying_covars])

        # Full Formula, adding the regression variable to the restricted formula
        if str(self.data.dtypes[regression_variable]) == 'category':
            formula = formula_restricted + f" + C({regression_variable})"
        else:
            formula = formula_restricted + f" + {regression_variable}"

        return formula_restricted, formula

    def get_results(self) -> Tuple[pd.DataFrame, Dict[str, List[str]], Dict[str, str]]:
        """
        Merge results into a dataFrame
        """
        if not self.run_complete:
            raise ValueError(f"No results: either the 'run' method was not called, or there was a problem running")

        result = pd.DataFrame.from_dict(self.results, orient='index')\
                             .reset_index(drop=False)\
                             .rename(columns={'index': 'Variable'})
        return result, self.warnings, self.errors

    def run_continuous(self, data, regression_variable, complete_case_idx, formula) -> Dict:
        result = dict()
        # Regress
        est = smf.glm(formula, data=data.loc[complete_case_idx], family=self.family).fit(use_t=self.use_t)
        # Save results if the regression converged
        if est.converged:
            result['Converged'] = True
            result['Beta'] = est.params[regression_variable]
            result['SE'] = est.bse[regression_variable]
            result['Variable_pvalue'] = est.pvalues[regression_variable]
            result['pvalue'] = result['Variable_pvalue']

        return result

    def run_binary(self, data, regression_variable, complete_case_idx, formula) -> Dict:
        result = dict()
        # Regress
        est = smf.glm(formula, data=data.loc[complete_case_idx], family=self.family).fit(use_t=self.use_t)
        # Check convergence
        # Save results if the regression converged
        if est.converged:
            result['Converged'] = True
            # Categorical-type RVs get a different name in the results, and aren't always at the end
            # (since categorical come before non-categorical)
            rv_keys = [k for k in est.params.keys() if regression_variable in k]
            try:
                assert len(rv_keys) == 1
                rv_key = rv_keys[0]
            except AssertionError:
                raise ValueError(f"Error extracting results for '{regression_variable}', try renaming the variable")
            result['Beta'] = est.params[rv_key]
            result['SE'] = est.bse[rv_key]
            result['Variable_pvalue'] = est.pvalues[rv_key]
            result['pvalue'] = result['Variable_pvalue']

        return result

    def run_categorical(self, data, regression_variable, complete_case_idx, formula, formula_restricted) -> Dict:
        result = dict()
        # Regress both models
        est = smf.glm(formula, data=data.loc[complete_case_idx], family=self.family).fit(use_t=self.use_t)
        est_restricted = smf.glm(formula_restricted, data=self.data.loc[complete_case_idx],
                                 family=self.family).fit(use_t=True)
        # Check convergence
        if est.converged & est_restricted.converged:
            result['Converged'] = True
            # Calculate Results
            lrdf = (est_restricted.df_resid - est.df_resid)
            lrstat = -2*(est_restricted.llf - est.llf)
            lr_pvalue = scipy.stats.chi2.sf(lrstat, lrdf)
            result['LRT_pvalue'] = lr_pvalue
            result['pvalue'] = result['LRT_pvalue']
            result['Diff_AIC'] = est.aic - est_restricted.aic

        return result

    def run(self):
        """Run a regression object, returning the results and logging any warnings/errors"""
        for rv_type, rv_list in self.regression_variables.items():
            click.echo(click.style(f"Running {len(rv_list):,} {rv_type} variables...", fg='green'))
            # TODO: Parallelize this loop
            for rv in rv_list:
                # Initialize result with placeholders
                self.results[rv] = self.get_default_result_dict()
                self.results[rv]['Variable_type'] = rv_type
                # Run in a try/except block to catch any errors specific to a regression variable
                try:
                    # Get complete case index and filter by min_n
                    complete_case_idx = self.get_complete_case_idx(self.data, rv)
                    N = len(complete_case_idx)
                    self.results[rv]['N'] = N
                    if N < self.min_n:
                        raise ValueError(f"too few complete observations (min_n filter: {N} < {self.min_n})")

                    # Check for covariates that do not vary (they get ignored)
                    varying_covars, warnings = self.check_covariate_values(complete_case_idx)
                    self.warnings[rv].extend(warnings)

                    # Take a copy of the required variables rather than operating directly on the stored data
                    data = self.data[[rv, self.outcome_variable] + varying_covars]

                    # Remove unused categories (warning when this occurs)
                    removed_cats = _remove_empty_categories(data)
                    if len(removed_cats) >= 1:
                        for extra_cat_var, extra_cats in removed_cats.items():
                            self.warnings[rv].append(f"'{str(extra_cat_var)}' had categories with no occurrences: "
                                                     f"{', '.join([str(c) for c in extra_cats])} "
                                                     f"after removing observations with missing values")

                    # Get the formulas
                    formula_restricted, formula = self.get_formulas(rv, varying_covars)

                    # Run Regression
                    if rv_type == 'continuous':
                        result = self.run_continuous(data, rv, complete_case_idx, formula)
                    elif rv_type == 'binary':  # Essentially same as continuous, except string used to key the results
                        result = self.run_binary(data, rv, complete_case_idx, formula)
                    elif rv_type == 'categorical':
                        result = self.run_categorical(data, rv, complete_case_idx, formula, formula_restricted)
                    else:
                        result = dict()
                    self.results[rv].update(result)

                except Exception as e:
                    self.errors[rv] = str(e)

            click.echo(click.style(f"Finished Running {len(rv_list):,} {rv_type} variables", fg='green'))
        self.run_complete = True
