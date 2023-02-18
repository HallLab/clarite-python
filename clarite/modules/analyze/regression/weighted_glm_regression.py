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

from clarite.internal.calculations import regTermTest
from clarite.internal.utilities import _get_dtypes, _remove_empty_categories
from clarite.modules.survey import SurveyDesignSpec, SurveyModel

from ..utils import fix_names, statsmodels_var_regex
from .glm_regression import GLMRegression

# GITHUB ISSUE #119: Regressions with Error after Multiprocessing release python > 3.8
multiprocessing.get_start_method("fork")


class WeightedGLMRegression(GLMRegression):
    """
    Statsmodels GLM Regression with adjustments for survey design.
    This class handles running a regression for each variable of interest and collecing results.
    The statistical adjustments (primarily the covariance calculation) are designed to match results when running with
    the R `survey` library.

    Notes
    -----
    * The family used is Gaussian for continuous outcomes or binomial(logit) for binary outcomes.
    * Covariates variables that are constant (after dropping rows due to missing data or applying subsets) produce
      warnings and are ignored
    * Rows missing a weight but not missing the tested variable will cause an error unless the `SurveyDesignSpec`
      specifies `drop_unweighted` as True (in which case those rows are dropped)
    * Categorical variables run with a survey design will not report Diff_AIC as it may not be possible to calculate
      it accurately

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
    survey_design_spec:
        A SurveyDesignSpec object is used to create SurveyDesign objects for each regression.
    min_n:
        Minimum number of complete-case observations (no NA values for outcome, covariates, variable, or weight)
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

    def __init__(
        self,
        data: pd.DataFrame,
        outcome_variable: str,
        regression_variables: List[str],
        covariates: Optional[List[str]],
        survey_design_spec: Optional[SurveyDesignSpec] = None,
        min_n: int = 200,
        report_categorical_betas: bool = False,
        standardize_data: bool = False,
        encoding: str = "additive",
        edge_encoding_info: Optional[pd.DataFrame] = None,
        process_num: Optional[int] = None,
    ):
        # survey_design_spec should actually not be None, but is a keyword for convenience
        if survey_design_spec is None:
            raise ValueError("A 'survey_design_spec' must be provided")

        # Base class __init__
        super().__init__(
            data=data,
            outcome_variable=outcome_variable,
            regression_variables=regression_variables,
            covariates=covariates,
            min_n=min_n,
            report_categorical_betas=report_categorical_betas,
            standardize_data=standardize_data,
            encoding=encoding,
            edge_encoding_info=edge_encoding_info,
            process_num=process_num,
        )

        # Custom init involving kwargs passed to this regression
        self.survey_design_spec = survey_design_spec

        # Add survey design info to the description
        self.description += "\n" + str(self.survey_design_spec)

        # Validate that survey design matches the data
        error = self.survey_design_spec.validate(data)
        if error is not None:
            raise ValueError(error)

    @staticmethod
    def get_default_result_dict(rv):
        return {
            "Variable": rv,
            "Weight": "",
            "Converged": False,
            "N": np.nan,
            "Beta": np.nan,
            "SE": np.nan,
            "Beta_pvalue": np.nan,
            "LRT_pvalue": np.nan,
            "Diff_AIC": np.nan,
            "pvalue": np.nan,
        }

    @staticmethod
    def _run_continuous_weighted(
        data, regression_variable, formula, survey_design_spec, family, use_t
    ) -> Dict:
        result = dict()
        # Get data based on the formula
        y, X = patsy.dmatrices(formula, data, return_type="dataframe", NA_action="drop")
        y = fix_names(y)
        X = fix_names(X)

        # Get survey design
        survey_design = survey_design_spec.get_survey_design(
            regression_variable, X.index
        )

        # Create and fit the model
        model = SurveyModel(
            design=survey_design,
            model_class=sm.GLM,
            init_args=dict(family=family),
            fit_args=dict(use_t=use_t),
        )
        model.fit(y=y, X=X)

        # Save results if the regression converged
        if model.result.converged:
            result["Converged"] = True
            rv_idx_list = [
                i
                for i, n in enumerate(X.columns)
                if re.match(statsmodels_var_regex(regression_variable), n)
            ]
            if len(rv_idx_list) != 1:
                raise ValueError(
                    f"Failed to find regression variable column in the results for {regression_variable}"
                )
            else:
                rv_idx = rv_idx_list[0]
            result["Beta"] = model.params[rv_idx]
            result["SE"] = model.stderr[rv_idx]
            # Calculate pvalue using a Two-sided t-test
            tval = np.abs(
                result["Beta"] / result["SE"]
            )  # T statistic is the absolute value of beta / SE
            dof = survey_design.get_dof(X)
            # Change SE to infinite and pvalue to 1 when dof < 1
            if dof < 1:
                result["SE"] = np.inf
                result["Beta_pvalue"] = 1.0
                result["pvalue"] = 1.0
            else:
                result["Beta_pvalue"] = scipy.stats.t.sf(tval, df=dof) * 2
                result["pvalue"] = result["Beta_pvalue"]

        return result

    @staticmethod
    def _run_categorical_weighted(
        data,
        regression_variable,
        formula,
        formula_restricted,
        survey_design_spec,
        family,
        use_t,
        report_categorical_betas,
    ) -> Dict:
        """
        See:
        Lumley, Thomas, and Alastair Scott. "Tests for regression models fitted to survey data."
        Australian & New Zealand Journal of Statistics 56.1 (2014): 1-14.
        """
        result = dict()

        # Regress full model
        y, X = patsy.dmatrices(formula, data, return_type="dataframe", NA_action="drop")
        y = fix_names(y)
        X = fix_names(X)
        # Get survey design
        survey_design = survey_design_spec.get_survey_design(
            regression_variable, X.index
        )
        model = SurveyModel(
            design=survey_design,
            model_class=sm.GLM,
            init_args=dict(family=family),
            fit_args=dict(use_t=use_t),
        )
        model.fit(y=y, X=X)

        # Regress restricted model
        y_restricted, X_restricted = patsy.dmatrices(
            formula_restricted, data, return_type="dataframe", NA_action="drop"
        )
        y_restricted = fix_names(y_restricted)
        X_restricted = fix_names(X_restricted)
        model_restricted = SurveyModel(
            design=survey_design,
            model_class=sm.GLM,
            init_args=dict(family=family),
            fit_args=dict(use_t=use_t),
        )
        model_restricted.fit(y=y_restricted, X=X_restricted)

        # Save results if the regression converged
        if model.result.converged & model_restricted.result.converged:
            result["Converged"] = True
            dof = survey_design.get_dof(X)
            lr_pvalue = regTermTest(
                full_model=model,
                restricted_model=model_restricted,
                ddf=dof,
                X_names=X.columns,
                var_name=regression_variable,
            )
            # Don't report AIC values for weighted categorical analysis since they may be incorrect
            if report_categorical_betas:
                rv_idx_list = [
                    i
                    for i, n in enumerate(X.columns)
                    if re.match(statsmodels_var_regex(regression_variable), n)
                ]
                for rv_idx in rv_idx_list:
                    beta = model.params[rv_idx]
                    se = model.stderr[rv_idx]
                    # Calculate pvalue using a Two-sided t-test
                    tval = np.abs(
                        beta / se
                    )  # T statistic is the absolute value of beta / SE
                    dof = survey_design.get_dof(X)
                    # Change SE to infinite and pvalue to 1 when dof < 1
                    if dof < 1:
                        se = np.inf
                        beta_pval = 1.0
                    else:
                        beta_pval = scipy.stats.t.sf(tval, df=dof) * 2
                    yield {
                        "Converged": True,
                        "LRT_pvalue": lr_pvalue,
                        "pvalue": lr_pvalue,
                        "Category": X.columns[rv_idx],
                        "Beta": beta,
                        "SE": se,
                        "Beta_pvalue": beta_pval,
                    }
            else:
                yield {"Converged": True, "LRT_pvalue": lr_pvalue, "pvalue": lr_pvalue}

        return result

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
                    self._run_weighted_rv(
                        rv,
                        rv_type,
                        self._get_rv_specific_data(rv),
                        self.outcome_variable,
                        self.covariates,
                        self.survey_design_spec,
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
                        self._run_weighted_rv,
                        zip(
                            rv_list,
                            repeat(rv_type),
                            [self._get_rv_specific_data(rv) for rv in rv_list],
                            repeat(self.outcome_variable),
                            repeat(self.covariates),
                            repeat(self.survey_design_spec),
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
    def _run_weighted_rv(
        cls,
        rv: str,
        rv_type: str,
        data: pd.DataFrame,
        outcome_variable: str,
        covariates: List[str],
        survey_design_spec: SurveyDesignSpec,
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
            # Get missing weight mask
            (
                weight_name,
                missing_weight_mask,
                warning,
            ) = survey_design_spec.check_missing_weights(data, rv)
            if warning is not None:
                warnings_list.append(warning)

            # Get complete case mask
            complete_case_mask = (
                ~data[[rv, outcome_variable] + covariates].isna().any(axis=1)
            )
            # If allowed (an error hasn't been raised) negate missing_weight_mask so True=keep to drop those
            # GITHUB ISSUE #117: Error type variable on Weight Regression with Clusters
            if missing_weight_mask is not None:
                complete_case_mask = complete_case_mask & ~missing_weight_mask

            # Count restricted rows
            restricted_rows = survey_design_spec.subset_array & complete_case_mask

            # Filter by min_n
            N = (restricted_rows).sum()
            if N < min_n:
                raise ValueError(
                    f"too few complete observations (min_n filter: {N} < {min_n})"
                )

            # Check for covariates that do not vary (they get ignored)
            varying_covars, warnings = cls._check_covariate_values(
                data, covariates, restricted_rows
            )
            warnings_list.extend(warnings)

            # Keep only restricted rows
            data = data.loc[restricted_rows]

            # Remove unused categories caused by dropping all occurrences of that value
            # during the above filtering (warning when this occurs)
            removed_cats = _remove_empty_categories(data)
            if len(removed_cats) >= 1:
                for extra_cat_var, extra_cats in removed_cats.items():
                    warning = (
                        f"'{str(extra_cat_var)}' had categories with no occurrences "
                        f"after removing observations with missing values"
                    )
                    if survey_design_spec.subset_count > 0:
                        warning += f" and applying the {survey_design_spec.subset_count} subset(s)"
                    warning += f": {', '.join([repr(c) for c in extra_cats])} "
                    warnings_list.append(warning)

            # Get the formulas
            # Restricted Formula, just outcome and covariates
            formula_restricted = f"Q('{outcome_variable}') ~ 1"
            if len(varying_covars) > 0:
                formula_restricted += " + "
                formula_restricted += " + ".join([f"Q('{v}')" for v in varying_covars])

            # Full Formula, adding the regression variable to the restricted formula
            formula = formula_restricted + f" + Q('{rv}" "')"

            # Update rv_type to the encoded type if it is a genotype
            if rv_type == "genotypes":
                """Need to update with encoded type"""
                rv_type = _get_dtypes(data[rv])[rv]

            # Run Regression
            if rv_type == "continuous":
                result = cls.get_default_result_dict(rv)
                result["Variable_type"] = rv_type
                result["Weight"] = weight_name
                result["N"] = N
                result.update(
                    cls._run_continuous_weighted(
                        data, rv, formula, survey_design_spec, family, use_t
                    )
                )
                result_list.append(result)
            elif (
                rv_type == "binary"
            ):  # The same calculation as for continuous variables
                result = cls.get_default_result_dict(rv)
                result["Variable_type"] = rv_type
                result["Weight"] = weight_name
                result["N"] = N
                result.update(
                    cls._run_continuous_weighted(
                        data, rv, formula, survey_design_spec, family, use_t
                    )
                )
                result_list.append(result)
            elif rv_type == "categorical":
                for r in cls._run_categorical_weighted(
                    data,
                    rv,
                    formula,
                    formula_restricted,
                    survey_design_spec,
                    family,
                    use_t,
                    report_categorical_betas,
                ):
                    result = cls.get_default_result_dict(rv)
                    result["Variable_type"] = rv_type
                    result["Weight"] = weight_name
                    result["N"] = N
                    result.update(r)
                    result_list.append(result)

        except Exception as e:
            error = str(e)
            if result is None:
                result_list = [cls.get_default_result_dict(rv)]

        return result_list, warnings_list, error
