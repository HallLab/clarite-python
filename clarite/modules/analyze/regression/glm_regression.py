import re
from typing import Dict, List, Tuple, Optional

import click
import numpy as np
import pandas as pd
import patsy
import scipy
import statsmodels.api as sm
from scipy.stats import stats

from clarite.internal.utilities import _remove_empty_categories

from .base import Regression
from ..utils import fix_names, statsmodels_var_regex


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
    data:
        The data to be analyzed, including the outcome, covariates, and any variables to be regressed.
    outcome_variable:
        The variable to be used as the output (y) of the regression
    covariates:
        The variables to be used as covariates.  Any variables in the DataFrames not listed as covariates are regressed.
    min_n:
        Minimum number of complete-case observations (no NA values for outcome, covariates, or variable)
        Defaults to 200
    report_categorical_betas: boolean
        False by default.
          If True, the results will contain one row for each categorical value (other than the reference category) and
          will include the beta value, standard error (SE), and beta pvalue for that specific category. The number of
          terms increases with the number of categories.
    standardize_data: boolean
        False by default.
          If True, numeric data will be standardized using z-scores before regression.  This will affect the beta values but not the pvalues.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        outcome_variable: str,
        covariates: Optional[List[str]] = None,
        min_n: int = 200,
        report_categorical_betas: bool = False,
        standardize_data: bool = False,
    ):
        """
        Parameters
        ----------
        data - pd.DataFrame
        outcome_variable - name of the outcome variable
        covariates - other variables to include in the regression formula

        Kwargs
        ______
        min_n - minimum number of observations (after discarding any with NA)
        report_categorical_betas - whether or not to report betas for individual categories
        """
        # base class init
        # This takes in minimal regression params (data, outcome_variable, covariates) and
        # initializes additional parameters (outcome dtype, regression variables, error, and warnings)
        super().__init__(
            data=data, outcome_variable=outcome_variable, covariates=covariates
        )

        # Custom init involving kwargs passed to this regression
        self.min_n = min_n
        self.report_categorical_betas = report_categorical_betas
        self.standardize_data = standardize_data

        # Ensure the data output type is compatible
        # Set 'self.family' and 'self.use_t' which are dependent on the outcome dtype
        if self.outcome_dtype == "categorical":
            raise NotImplementedError(
                "Categorical Outcomes are not yet supported for this type of regression."
            )
        elif self.outcome_dtype == "continuous":
            self.description += (
                f"Continuous Outcome (family = Gaussian): '{self.outcome_variable}'"
            )
            self.family = sm.families.Gaussian(link=sm.families.links.identity())
            self.use_t = True
        elif self.outcome_dtype == "binary":
            # Use the order according to the categorical
            counts = self.data[self.outcome_variable].value_counts().to_dict()
            categories = self.data[self.outcome_variable].cat.categories
            codes, categories = zip(*enumerate(categories))
            self.data[self.outcome_variable].replace(categories, codes, inplace=True)
            self.description += (
                f"Binary Outcome (family = Binomial): '{self.outcome_variable}'\n"
                f"\t{counts[categories[0]]:,} occurrences of '{categories[0]}' coded as 0\n"
                f"\t{counts[categories[1]]:,} occurrences of '{categories[1]}' coded as 1"
            )
            self.family = sm.families.Binomial(link=sm.families.links.logit())
            self.use_t = False
        else:
            raise ValueError(
                "The outcome variable's type could not be determined.  Please report this error."
            )

        # Log missing outcome values
        na_outcome_count = self.data[self.outcome_variable].isna().sum()
        self.description += f"\nUsing {len(self.data) - na_outcome_count:,} of {len(self.data):,} observations"
        if na_outcome_count > 0:
            self.description += (
                f"\n\t{na_outcome_count:,} are missing a value for the outcome variable"
            )

        # Finish updating description
        self.description += f"\nRegressing {sum([len(v) for v in self.regression_variables.values()]):,} variables"
        for k, v in self.regression_variables.items():
            self.description += f"\n\t{len(v):,} {k} variables"

    @staticmethod
    def get_default_result_dict(rv):
        return {
            "Variable": rv,
            "Converged": False,
            "N": np.nan,
            "Beta": np.nan,
            "SE": np.nan,
            "Beta_pvalue": np.nan,
            "LRT_pvalue": np.nan,
            "Diff_AIC": np.nan,
            "pvalue": np.nan,
            "Weight": None,
        }

    def _get_formulas(self, regression_variable, varying_covars) -> Tuple[str, str]:
        # Restricted Formula, just outcome and covariates
        formula_restricted = f"Q('{self.outcome_variable}') ~ 1"
        if len(varying_covars) > 0:
            formula_restricted += " + "
            formula_restricted += " + ".join([f"Q('{v}')" for v in varying_covars])

        # Full Formula, adding the regression variable to the restricted formula
        formula = formula_restricted + f" + Q('{regression_variable}" "')"

        return formula_restricted, formula

    def _process_formula(self, formula, data) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Use patsy to process the formula with quoted variable names, but return with the original names.
        Standardize data if enabled.
        """
        y, X = patsy.dmatrices(formula, data, return_type="dataframe", NA_action="drop")
        y = fix_names(y)
        X = fix_names(X)
        if self.standardize_data:
            # Transform numeric columns in X, skipping the intercept (first column)
            X[[c for c in X.columns if c != "Intercept"]] = (
                X[[c for c in X.columns if c != "Intercept"]]
                .select_dtypes(include=[np.number])
                .dropna()
                .apply(stats.zscore)
            )
            if self.outcome_dtype == "continuous":
                y = stats.zscore(y)
        return y, X

    def get_results(self) -> pd.DataFrame:
        """
        Get regression results if `run` has already been called

        Returns
        -------
        result: pd.DataFrame
        """
        if not self.run_complete:
            raise ValueError(
                "No results: either the 'run' method was not called, or there was a problem running"
            )
        self._log_errors_and_warnings()
        result = pd.DataFrame(self.results).astype({"N": pd.Int64Dtype()})

        # Add "Outcome" and set the index
        result["Outcome"] = self.outcome_variable
        if self.report_categorical_betas:
            result = result.set_index(["Variable", "Outcome", "Category"]).sort_values(
                ["pvalue", "Beta_pvalue"]
            )
        else:
            result = result.set_index(["Variable", "Outcome"]).sort_values(["pvalue"])

        # Order columns
        column_order = [
            "Variable_type",
            "Weight",
            "Converged",
            "N",
            "Beta",
            "SE",
            "Beta_pvalue",
            "LRT_pvalue",
            "Diff_AIC",
            "pvalue",
        ]
        result = result[column_order]

        return result

    def _run_continuous(self, data, regression_variable, formula) -> Dict:
        result = dict()
        # Regress
        y, X = self._process_formula(formula, data)
        est = sm.GLM(y, X, family=self.family).fit(use_t=self.use_t)
        # Save results if the regression converged
        if est.converged:
            result["Converged"] = True
            result["Beta"] = est.params[regression_variable]
            result["SE"] = est.bse[regression_variable]
            result["Beta_pvalue"] = est.pvalues[regression_variable]
            result["pvalue"] = result["Beta_pvalue"]

        return result

    def _run_binary(self, data, regression_variable, formula) -> Dict:
        result = dict()
        # Regress
        y, X = self._process_formula(formula, data)
        est = sm.GLM(y, X, family=self.family).fit(use_t=self.use_t)
        # Check convergence
        # Save results if the regression converged
        if est.converged:
            result["Converged"] = True
            # Categorical-type RVs get a different name in the results, and aren't always at the end
            # (since categorical come before non-categorical)
            rv_keys = [
                k
                for k in est.params.keys()
                if re.match(statsmodels_var_regex(regression_variable), k)
            ]
            try:
                assert len(rv_keys) == 1
                rv_key = rv_keys[0]
            except AssertionError:
                raise ValueError(
                    f"Error extracting results for '{regression_variable}', try renaming the variable"
                )
            result["Beta"] = est.params[rv_key]
            result["SE"] = est.bse[rv_key]
            result["Beta_pvalue"] = est.pvalues[rv_key]
            result["pvalue"] = result["Beta_pvalue"]

        return result

    def _run_categorical(self, data, formula, formula_restricted) -> Dict:
        result = dict()
        # Regress both models
        y, X = self._process_formula(formula, data)
        est = sm.GLM(y, X, family=self.family).fit(use_t=self.use_t)
        y_restricted, X_restricted = self._process_formula(formula_restricted, data)
        est_restricted = sm.GLM(y_restricted, X_restricted, family=self.family).fit(
            use_t=True
        )
        # Check convergence
        if est.converged & est_restricted.converged:
            # Calculate Results
            lrdf = est_restricted.df_resid - est.df_resid
            lrstat = -2 * (est_restricted.llf - est.llf)
            lr_pvalue = scipy.stats.chi2.sf(lrstat, lrdf)
            if self.report_categorical_betas:
                param_names = set(est.bse.index) - set(est_restricted.bse.index)
                # The restricted model shouldn't have extra terms, unless there is some case we have overlooked
                assert len(set(est_restricted.bse.index) - set(est.bse.index)) == 0
                for param_name in param_names:
                    yield {
                        "Converged": True,
                        "Category": param_name,
                        "Beta": est.params[param_name],
                        "SE": est.bse[param_name],
                        "Beta_pvalue": est.pvalues[param_name],
                        "LRT_pvalue": lr_pvalue,
                        "pvalue": lr_pvalue,
                        "Diff_AIC": est.aic - est_restricted.aic,
                    }
            else:
                # Only return the LRT result
                yield {
                    "Converged": True,
                    "LRT_pvalue": lr_pvalue,
                    "pvalue": lr_pvalue,
                    "Diff_AIC": est.aic - est_restricted.aic,
                }

    def run(self):
        """Run a regression object, returning the results and logging any warnings/errors"""
        for rv_type, rv_list in self.regression_variables.items():
            click.echo(
                click.style(
                    f"Running {len(rv_list):,} {rv_type} variables...", fg="green"
                )
            )
            # TODO: Parallelize this loop
            for rv in rv_list:
                # Run in a try/except block to catch any errors specific to a regression variable
                try:
                    # Take a copy of the data (ignoring other RVs)
                    keep_columns = [rv, self.outcome_variable] + self.covariates
                    data = self.data[keep_columns]

                    # Get complete case mask and filter by min_n
                    complete_case_mask = (
                        ~data[[rv, self.outcome_variable] + self.covariates]
                        .isna()
                        .any(axis=1)
                    )
                    N = complete_case_mask.sum()
                    if N < self.min_n:
                        raise ValueError(
                            f"too few complete observations (min_n filter: {N} < {self.min_n})"
                        )

                    # Check for covariates that do not vary (they get ignored)
                    varying_covars, warnings = self._check_covariate_values(
                        complete_case_mask
                    )
                    self.warnings[rv].extend(warnings)

                    # Remove unused categories (warning when this occurs)
                    removed_cats = _remove_empty_categories(data)
                    if len(removed_cats) >= 1:
                        for extra_cat_var, extra_cats in removed_cats.items():
                            self.warnings[rv].append(
                                f"'{str(extra_cat_var)}' had categories with no occurrences: "
                                f"{', '.join([str(c) for c in extra_cats])} "
                                f"after removing observations with missing values"
                            )

                    # Get the formulas
                    formula_restricted, formula = self._get_formulas(rv, varying_covars)

                    # Apply the complete_case_mask to the data to ensure categorical models use the same data in the LRT
                    data = data.loc[complete_case_mask]

                    # Run Regression
                    if rv_type == "continuous":
                        result = self.get_default_result_dict(rv)
                        result["Variable_type"] = rv_type
                        result["N"] = N
                        result.update(self._run_continuous(data, rv, formula))
                        self.results.append(result)
                    elif (
                        rv_type == "binary"
                    ):  # Essentially same as continuous, except string used to key the results
                        # Initialize result with placeholders
                        result = self.get_default_result_dict(rv)
                        result["Variable_type"] = rv_type
                        result["N"] = N
                        result.update(self._run_binary(data, rv, formula))
                        self.results.append(result)
                    elif rv_type == "categorical":
                        for r in self._run_categorical(
                            data, formula, formula_restricted
                        ):
                            # Initialize result with placeholders
                            result = self.get_default_result_dict(rv)
                            result["Variable_type"] = rv_type
                            result["N"] = N
                            result.update(r)
                            self.results.append(result)

                except Exception as e:
                    self.errors[rv] = str(e)
                    self.results.append(result)

            click.echo(
                click.style(
                    f"\tFinished Running {len(rv_list):,} {rv_type} variables",
                    fg="green",
                )
            )
        self.run_complete = True
