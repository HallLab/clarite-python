import re
from typing import Dict, List, Tuple

import click
import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf

from clarite.internal.utilities import _remove_empty_categories

from .base import Regression


class GLMRegression(Regression):
    """
    Statsmodels GLM Regression.
    This class handles running a regression for each variable of interest and collecting results.

    Regression Methods
    ------------------
    Binary variables
        Treated as continuous features, with values of 0 and 1 (the larger value in the original data is encoded as 1).
    Categorical variables
        The results of a likelihood ratio test are used to calculate a pvalue.  No Beta or SE values are reported.
    Continuous variables
        A GLM is used to obtain Beta, SE, and pvalue results.

    Notes
    -----
    * The family used is either Gaussian (continuous outcomes) or binomial(logit) for binary outcomes.
    * Covariates variables that are constant produce warnings and are ignored
    * The dataset is subset to drop missing values, and the same dataset is used for both models in the LRT

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

        # Ensure the data output type is compatible
        # Set 'self.family' and 'self.use_t' which are dependent on the outcome dtype
        if self.outcome_dtype == 'categorical':
            raise NotImplementedError("Categorical Phenotypes are not yet supported for this type of regression.")
        elif self.outcome_dtype == 'continuous':
            self.description += f"Continuous Outcome (family = Gaussian): '{self.outcome_variable}'"
            self.family = sm.families.Gaussian(link=sm.families.links.identity())
            self.use_t = True
        elif self.outcome_dtype == 'binary':
            # Use the order according to the categorical
            counts = self.data[self.outcome_variable].value_counts().to_dict()
            categories = self.data[self.outcome_variable].cat.categories
            codes, categories = zip(*enumerate(categories))
            self.data[self.outcome_variable].replace(categories, codes, inplace=True)
            self.description += f"Binary Outcome (family = Binomial): '{self.outcome_variable}'\n" \
                                f"\t{counts[categories[0]]:,} occurrences of '{categories[0]}' coded as 0\n" \
                                f"\t{counts[categories[1]]:,} occurrences of '{categories[1]}' coded as 1"
            self.family = sm.families.Binomial(link=sm.families.links.logit())
            self.use_t = False
        else:
            raise ValueError("The outcome variable's type could not be determined.  Please report this error.")

        # Log missing outcome values
        na_outcome_count = self.data[self.outcome_variable].isna().sum()
        self.description += f"\nUsing {len(self.data) - na_outcome_count:,} of {len(self.data):,} observations"
        if na_outcome_count > 0:
            self.description += f"\n\t{na_outcome_count:,} are missing a value for the outcome variable"

        # Finish updating description
        self.description += f"\nRegressing {len(self.results):,} variables"
        for k, v in self.regression_variables.items():
            self.description += f"\n\t{len(v):,} {k} variables"

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

    def get_complete_case_mask(self, data, regression_variable):
        """
        Get boolean mask of observations that are not missing in the test variable, the outcome variable, and covariates
        Parameters
        ----------
        data
        regression_variable

        Returns
        -------
        Pd.Series
            Boolean series with True = complete case and False = missing one or more values
        """
        return ~data[[regression_variable, self.outcome_variable] + self.covariates].isna().any(axis=1)

    def check_covariate_values(self, keep_row_mask) -> Tuple[List[str], List[str]]:
        """Remove covariates that do not vary, warning when this occurs"""
        warnings = []
        unique_values = self.data.loc[keep_row_mask, self.covariates].nunique()
        varying_covars = list(unique_values[unique_values > 1].index.values)
        non_varying_covars = list(unique_values[unique_values <= 1].index.values)
        if len(non_varying_covars) > 0:
            warnings.append(f"non-varying covariates(s): {', '.join(non_varying_covars)}")
        return varying_covars, warnings

    def get_formulas(self, regression_variable, varying_covars) -> Tuple[str, str]:
        # Restricted Formula, just outcome and covariates
        formula_restricted = f"{self.outcome_variable} ~ 1"
        if len(varying_covars) > 0:
            formula_restricted += " + "
            formula_restricted += " + ".join(varying_covars)

        # Full Formula, adding the regression variable to the restricted formula
        formula = formula_restricted + f" + {regression_variable}"

        return formula_restricted, formula

    def get_results(self) -> pd.DataFrame:
        """
        Get regression results if `run` has already been called

        Returns
        -------
        result: pd.DataFrame
            Results DataFrame with these columns:
            ['variable_type', 'N', 'beta', 'SE', 'var_pvalue', 'LRT_pvalue', 'diff_AIC', 'pvalue']
        """
        if not self.run_complete:
            raise ValueError("No results: either the 'run' method was not called, or there was a problem running")

        # Log errors
        if len(self.errors) == 0:
            click.echo(click.style("0 regression variables had an error", fg='green'))
        elif len(self.errors) > 0:
            click.echo(click.style(f"{len(self.errors):,} regression variables had an error", fg='red'))
            for rv, error in self.errors.items():
                click.echo(click.style(f"\t{rv} = NULL due to: {error}", fg='red'))

        # Log warnings
        for rv, warning_list in self.warnings.items():
            if len(warning_list) > 0:
                click.echo(click.style(f"{rv} had warnings:", fg='yellow'))
                for warning in warning_list:
                    click.echo(click.style(f"\t{warning}", fg='yellow'))

        result = pd.DataFrame.from_dict(self.results, orient='index')\
                             .reset_index(drop=False)\
                             .rename(columns={'index': 'Variable'})\
                             .astype({"N": pd.Int64Dtype()})  # b/c N isn't checked when weights are missing

        return result

    def run_continuous(self, data, regression_variable, formula) -> Dict:
        result = dict()
        # Regress
        est = smf.glm(formula, data=data, family=self.family).fit(use_t=self.use_t)
        # Save results if the regression converged
        if est.converged:
            result['Converged'] = True
            result['Beta'] = est.params[regression_variable]
            result['SE'] = est.bse[regression_variable]
            result['Variable_pvalue'] = est.pvalues[regression_variable]
            result['pvalue'] = result['Variable_pvalue']

        return result

    def run_binary(self, data, regression_variable, formula) -> Dict:
        result = dict()
        # Regress
        est = smf.glm(formula, data=data, family=self.family).fit(use_t=self.use_t)
        # Check convergence
        # Save results if the regression converged
        if est.converged:
            result['Converged'] = True
            # Categorical-type RVs get a different name in the results, and aren't always at the end
            # (since categorical come before non-categorical)
            rv_keys = [k for k in est.params.keys()
                       if re.match(fr"^{regression_variable}(\[T\..*\])?$", k)]  # <regression_variable>[T.<category>]
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

    def run_categorical(self, data, formula, formula_restricted) -> Dict:
        result = dict()
        # Regress both models
        est = smf.glm(formula, data=data, family=self.family).fit(use_t=self.use_t)
        est_restricted = smf.glm(formula_restricted, data=data,
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
                    # Take a copy of the data (ignoring other RVs)
                    keep_columns = [rv, self.outcome_variable] + self.covariates
                    data = self.data[keep_columns]

                    # Get complete case mask and filter by min_n
                    complete_case_mask = self.get_complete_case_mask(data, rv)
                    N = complete_case_mask.sum()
                    self.results[rv]['N'] = N
                    if N < self.min_n:
                        raise ValueError(f"too few complete observations (min_n filter: {N} < {self.min_n})")

                    # Check for covariates that do not vary (they get ignored)
                    varying_covars, warnings = self.check_covariate_values(complete_case_mask)
                    self.warnings[rv].extend(warnings)

                    # Remove unused categories (warning when this occurs)
                    removed_cats = _remove_empty_categories(data)
                    if len(removed_cats) >= 1:
                        for extra_cat_var, extra_cats in removed_cats.items():
                            self.warnings[rv].append(f"'{str(extra_cat_var)}' had categories with no occurrences: "
                                                     f"{', '.join([str(c) for c in extra_cats])} "
                                                     f"after removing observations with missing values")

                    # Get the formulas
                    formula_restricted, formula = self.get_formulas(rv, varying_covars)

                    # Apply the complete_case_mask to the data to ensure categorical models use the same data in the LRT
                    data = data.loc[complete_case_mask]

                    # Run Regression
                    if rv_type == 'continuous':
                        result = self.run_continuous(data, rv, formula)
                    elif rv_type == 'binary':  # Essentially same as continuous, except string used to key the results
                        result = self.run_binary(data, rv, formula)
                    elif rv_type == 'categorical':
                        result = self.run_categorical(data, formula, formula_restricted)
                    else:
                        result = dict()
                    self.results[rv].update(result)

                except Exception as e:
                    self.errors[rv] = str(e)

            click.echo(click.style(f"\tFinished Running {len(rv_list):,} {rv_type} variables", fg='green'))
        self.run_complete = True
