import multiprocessing
import re
from itertools import repeat
from typing import Dict, List, Optional, Tuple

import click
import numpy as np
import pandas as pd
import patsy
import scipy
import statsmodels.api as sm
from pandas_genomics import GenotypeDtype
from scipy.stats import stats

from clarite.internal.utilities import _get_dtypes, _remove_empty_categories

from ..utils import fix_names, statsmodels_var_regex
from .base import Regression

# GITHUB ISSUE #119: Regressions with Error after Multiprocessing release python > 3.8
multiprocessing.get_start_method("fork")


class GLMRegression(Regression):
    """
    Statsmodels GLM Regression.
    This class handles running a regression for each variable of interest and collecting results.

    Notes
    -----
    * The family used is either Gaussian (continuous outcomes) or binomial(logit) for binary outcomes.
    * Covariates variables that are constant produce warnings and are ignored
    * The dataset is subset to drop missing values, and the same dataset is used for both models in the LRT

    *Regression Methods*

    Binary variables
        Treated as continuous features, with values of 0 and 1 (the larger value in the original data is encoded as 1).
    Categorical variables
        The results of a likelihood ratio test are used to calculate a pvalue.  No Beta or SE values are reported.
    Continuous variables
        A GLM is used to obtain Beta, SE, and pvalue results.

    Parameters
    ----------
    data:
        The data to be analyzed, including the outcome, covariates, and any variables to be regressed.
    outcome_variable:
        The variable to be used as the output (y) of the regression
    regression_variables:
        List of regression variables to be used as input
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
          If True, numeric data will be standardized using z-scores before regression.
          This will affect the beta values and standard error, but not the pvalues.
    encoding: str, default "additive"
        Encoding method to use for any genotype data.  One of {'additive', 'dominant', 'recessive', 'codominant', or 'weighted'}
    edge_encoding_info: Optional pd.DataFrame, default None
        If edge encoding is used, this must be provided.  See Pandas-Genomics documentation on edge encodings.
    process_num: Optional[int]
        Number of processes to use when running the analysis, default is None (use the number of cores)
    """

    KNOWN_ENCODINGS = {"additive", "dominant", "recessive", "codominant", "edge"}

    def __init__(
        self,
        data: pd.DataFrame,
        outcome_variable: str,
        regression_variables: List[str],
        covariates: Optional[List[str]] = None,
        min_n: int = 200,
        report_categorical_betas: bool = False,
        standardize_data: bool = False,
        encoding: str = "additive",
        edge_encoding_info: Optional[pd.DataFrame] = None,
        process_num: Optional[int] = None,
    ):
        # base class init
        # This takes in minimal regression params (data, outcome_variable, covariates) and
        # initializes additional parameters (outcome dtype, regression variables, error, and warnings)
        super().__init__(
            data=data,
            outcome_variable=outcome_variable,
            regression_variables=regression_variables,
            covariates=covariates,
        )

        # Custom init involving kwargs passed to this regression
        self.min_n = min_n
        self.report_categorical_betas = report_categorical_betas
        self.standardize_data = standardize_data
        if process_num is None:
            process_num = multiprocessing.cpu_count()
        self.process_num = process_num
        if encoding not in self.KNOWN_ENCODINGS:
            raise ValueError(f"Genotypes provided with unknown 'encoding': {encoding}")
        elif encoding == "edge" and edge_encoding_info is None:
            raise ValueError(
                "'edge_encoding_info' must be provided when using edge encoding"
            )
        else:
            self.encoding = encoding
            self.edge_encoding_info = edge_encoding_info

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
            # GITHUB ISSUES #115: Keep control as 0 and case as 1
            if categories[0] == "Case" and categories[1] == "Control":
                categories = sorted(categories, reverse=True)

            # TODO: Allow only 0/1 or Control/Case entries | Other entries generate a warning

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

        # Standardize continuous variables in the data if needed
        # Use ddof=1 in the zscore calculation (used for StdErr) to match R
        if self.standardize_data:
            if self.outcome_dtype == "continuous":
                self.data[self.outcome_variable] = stats.zscore(
                    self.data[self.outcome_variable], nan_policy="omit", ddof=1
                )
            continuous_rvs = self.regression_variables["continuous"]
            self.data[continuous_rvs] = stats.zscore(
                self.data[continuous_rvs], nan_policy="omit", ddof=1
            )
            continuous_covars = [
                rv
                for rv, rv_type in self.covariate_types.items()
                if rv_type == "continuous"
            ]
            self.data[continuous_covars] = stats.zscore(
                self.data[continuous_covars], nan_policy="omit", ddof=1
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
            # If there were no categorical variables (probably a mistake to set this option) the "Category" column will be missing.
            if "Category" not in result.columns:
                result["Category"] = None
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

        # Update datatypes
        result["Weight"] = result["Weight"].fillna("None").astype("category")

        return result

    @staticmethod
    def _run_continuous(data, regression_variable, formula, family, use_t) -> Dict:
        result = dict()
        # Regress
        y, X = patsy.dmatrices(formula, data, return_type="dataframe", NA_action="drop")
        y = fix_names(y)
        X = fix_names(X)
        est = sm.GLM(y, X, family=family).fit(use_t=use_t)
        # Save results if the regression converged
        if est.converged:
            result["Converged"] = True
            result["Beta"] = est.params[regression_variable]
            result["SE"] = est.bse[regression_variable]
            result["Beta_pvalue"] = est.pvalues[regression_variable]
            result["pvalue"] = result["Beta_pvalue"]

        return result

    @staticmethod
    def _run_binary(data, regression_variable, formula, family, use_t) -> Dict:
        result = dict()
        # Regress
        y, X = patsy.dmatrices(formula, data, return_type="dataframe", NA_action="drop")
        y = fix_names(y)
        X = fix_names(X)
        est = sm.GLM(y, X, family=family).fit(use_t=use_t)
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

    @staticmethod
    def _run_categorical(
        data, formula, formula_restricted, family, use_t, report_categorical_betas
    ) -> Dict:
        # Regress both models
        y, X = patsy.dmatrices(formula, data, return_type="dataframe", NA_action="drop")
        y = fix_names(y)
        X = fix_names(X)
        est = sm.GLM(y, X, family=family).fit(use_t=use_t)

        y_restricted, X_restricted = patsy.dmatrices(
            formula_restricted, data, return_type="dataframe", NA_action="drop"
        )
        y_restricted = fix_names(y_restricted)
        X_restricted = fix_names(X_restricted)
        est_restricted = sm.GLM(y_restricted, X_restricted, family=family).fit(
            use_t=use_t
        )
        # Check convergence
        if est.converged & est_restricted.converged:
            # Calculate Results
            lrdf = est_restricted.df_resid - est.df_resid
            lrstat = -2 * (est_restricted.llf - est.llf)
            lr_pvalue = scipy.stats.chi2.sf(lrstat, lrdf)
            if report_categorical_betas:
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

    def _get_rv_specific_data(self, rv: str):
        """Select the data relevant to performing a regression on a given variable, encoding genotypes if needed"""
        data = self.data[[rv, self.outcome_variable] + self.covariates].copy()
        # Encode any genotype data
        has_genotypes = False
        for dt in data.dtypes:
            if GenotypeDtype.is_dtype(dt):
                has_genotypes = True
                break
        if has_genotypes:
            if self.encoding == "additive":
                data = data.genomics.encode_additive()
            elif self.encoding == "dominant":
                data = data.genomics.encode_dominant()
            elif self.encoding == "recessive":
                data = data.genomics.encode_recessive()
            elif self.encoding == "codominant":
                data = data.genomics.encode_codominant()
            elif self.encoding == "edge":
                data = data.genomics.encode_edge(self.edge_encoding_info)
        return data

    def run(self):
        """Run a regression object, returning the results and logging any warnings/errors"""
        for rv_type, rv_list in self.regression_variables.items():
            if len(rv_list) == 0:
                click.echo(click.style(f"No {rv_type} variables to run...", fg="green"))
                continue
            else:
                click.echo(
                    click.style(
                        f"Running {len(rv_list):,} {rv_type} variables using {self.process_num} processes...",
                        fg="green",
                    )
                )

            # TODO: Error on multiprocess after update to Python > 3.8
            self.process_num = 1

            if self.process_num == 1:
                run_result = [
                    self._run_rv(
                        rv,
                        rv_type,
                        self._get_rv_specific_data(rv),
                        self.outcome_variable,
                        self.covariates,
                        self.min_n,
                        self.family,
                        self.use_t,
                        self.report_categorical_betas,
                    )
                    for rv in rv_list
                ]
            else:
                with multiprocessing.Pool(processes=self.process_num) as pool:
                    run_result = pool.starmap(
                        self._run_rv,
                        zip(
                            rv_list,
                            repeat(rv_type),
                            [self._get_rv_specific_data(rv) for rv in rv_list],
                            repeat(self.outcome_variable),
                            repeat(self.covariates),
                            repeat(self.min_n),
                            repeat(self.family),
                            repeat(self.use_t),
                            repeat(self.report_categorical_betas),
                        ),
                    )

            for rv, rv_result in zip(rv_list, run_result):
                results, warnings, error = rv_result
                self.results.extend(results)  # Merge lists into one list
                self.warnings[rv] = warnings
                if error is not None:
                    self.errors[rv] = error

            click.echo(
                click.style(
                    f"\tFinished Running {len(rv_list):,} {rv_type} variables",
                    fg="green",
                )
            )
        self.run_complete = True

    @classmethod
    def _run_rv(
        cls,
        rv: str,
        rv_type: str,
        data: pd.DataFrame,
        outcome_variable: str,
        covariates: List[str],
        min_n: int,
        family: str,
        use_t: bool,
        report_categorical_betas: bool,
    ) -> Tuple[List[dict], List[str], str]:  # results, warnings, errors
        # Initialize return values
        result_list = []
        warnings_list = []
        error = None

        # Must define result to catch errors outside running individual variables
        result = None

        # Run in a try/except block to catch any errors specific to a regression variable
        try:
            # Get complete case mask and filter by min_n
            complete_case_mask = ~data.isna().any(axis=1)
            N = complete_case_mask.sum()
            if N < min_n:
                raise ValueError(
                    f"too few complete observations (min_n filter: {N} < {min_n})"
                )

            # Check for covariates that do not vary (they get ignored)
            varying_covars, warnings = cls._check_covariate_values(
                data, covariates, complete_case_mask
            )
            warnings_list.extend(warnings)

            # GIT ISSUES 116: Regression matrix with empty categories
            # (Moved after to clear NAN)
            # Remove unused categories (warning when this occurs)
            # removed_cats = _remove_empty_categories(data)
            # if len(removed_cats) >= 1:
            #     for extra_cat_var, extra_cats in removed_cats.items():
            #         warnings_list.append(
            #             f"'{str(extra_cat_var)}' had categories with no occurrences: "
            #             f"{', '.join([str(c) for c in extra_cats])} "
            #             f"after removing observations with missing values"
            #         )

            # Get the formulas
            # Restricted Formula, just outcome and covariates
            formula_restricted = f"Q('{outcome_variable}') ~ 1"
            if len(varying_covars) > 0:
                formula_restricted += " + "
                formula_restricted += " + ".join([f"Q('{v}')" for v in varying_covars])

            # Full Formula, adding the regression variable to the restricted formula
            formula = formula_restricted + f" + Q('{rv}" "')"

            # Apply the complete_case_mask to the data to ensure categorical models use the same data in the LRT
            data = data.loc[complete_case_mask]

            # GIT ISSUES 116: Regression matrix with empty categories
            # Remove unused categories (warning when this occurs)
            removed_cats = _remove_empty_categories(data)
            if len(removed_cats) >= 1:
                for extra_cat_var, extra_cats in removed_cats.items():
                    warnings_list.append(
                        f"'{str(extra_cat_var)}' had categories with no occurrences: "
                        f"{', '.join([str(c) for c in extra_cats])} "
                        f"after removing observations with missing values"
                    )

            # Update rv_type to the encoded type if it is a genotype
            if rv_type == "genotypes":
                """Need to update with encoded type"""
                rv_type = _get_dtypes(data[rv])[rv]

            # Run Regression
            if rv_type == "continuous":
                result = cls.get_default_result_dict(rv)
                result["Variable_type"] = rv_type
                result["N"] = N
                result.update(cls._run_continuous(data, rv, formula, family, use_t))
                result_list.append(result)
            elif (
                rv_type == "binary"
            ):  # Essentially same as continuous, except string used to key the results
                # Initialize result with placeholders
                result = cls.get_default_result_dict(rv)
                result["Variable_type"] = rv_type
                result["N"] = N
                result.update(cls._run_binary(data, rv, formula, family, use_t))
                result_list.append(result)
            elif rv_type == "categorical":
                for r in cls._run_categorical(
                    data,
                    formula,
                    formula_restricted,
                    family,
                    use_t,
                    report_categorical_betas,
                ):
                    # Initialize result with placeholders
                    result = cls.get_default_result_dict(rv)
                    result["Variable_type"] = rv_type
                    result["N"] = N
                    result.update(r)
                    result_list.append(result)

        except Exception as e:
            error = str(e)
            if result is None:
                result_list = [cls.get_default_result_dict(rv)]

        return result_list, warnings_list, error
