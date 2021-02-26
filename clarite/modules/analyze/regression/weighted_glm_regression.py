import re
from typing import Optional, Dict, List

import click
import numpy as np
import scipy
import pandas as pd
import statsmodels.api as sm

from .glm_regression import GLMRegression
from clarite.modules.survey import SurveyDesignSpec, SurveyModel
from clarite.internal.calculations import regTermTest
from clarite.internal.utilities import _remove_empty_categories
from ..utils import statsmodels_var_regex


class WeightedGLMRegression(GLMRegression):
    """
    Statsmodels GLM Regression with adjustments for survey design.
    This class handles running a regression for each variable of interest and collecing results.
    The statistical adjustments (primarily the covariance calculation) are designed to match results when running with
    the R `survey` library.

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
    * The family used is Gaussian for continuous outcomes or binomial(logit) for binary outcomes.
    * Covariates variables that are constant (after dropping rows due to missing data or applying subsets) produce
      warnings and are ignored
    * Rows missing a weight but not missing the tested variable will cause an error unless the `SurveyDesignSpec`
      specifies `drop_unweighted` as True (in which case those rows are dropped)
    * Categorical variables run with a survey design will not report Diff_AIC as it may not be possible to calculate
      it accurately

    Parameters
    ----------
    data:
        The data to be analyzed, including the outcome, covariates, and any variables to be regressed.
    outcome_variable:
        The variable to be used as the output (y) of the regression
    covariates:
        The variables to be used as covariates.  Any variables in the DataFrames not listed as covariates are regressed.
    survey_design_spec:
        A SurveyDesignSpec object is used to create SurveyDesign objects for each regression.
    min_n:
        Minimum number of complete-case observations (no NA values for outcome, covariates, variable, or weight)
        Defaults to 200
    cov_method:
        Covariance calculation method (if survey_design_spec is passed in).  'stata' by default.
        Warning: `jackknife` is untested and may not be accurate
    """

    def __init__(
        self,
        data: pd.DataFrame,
        outcome_variable: str,
        covariates: Optional[List[str]],
        survey_design_spec: Optional[SurveyDesignSpec] = None,
        min_n: int = 200,
        cov_method: Optional[str] = "stata",
    ):
        # survey_design_spec should actually not be None, but is a keyword for convenience
        if survey_design_spec is None:
            raise ValueError("A 'survey_design_spec' must be provided")

        # Base class __init__
        super().__init__(
            data=data,
            outcome_variable=outcome_variable,
            covariates=covariates,
            min_n=min_n,
        )

        # Custom init involving kwargs passed to this regression
        self.survey_design_spec = survey_design_spec
        self.cov_method = cov_method

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
            "Variable_pvalue": np.nan,
            "LRT_pvalue": np.nan,
            "Diff_AIC": np.nan,
            "pvalue": np.nan,
        }

    def _run_continuous_weighted(self, data, regression_variable, formula) -> Dict:
        result = dict()
        # Get data based on the formula
        y, X = self._process_formula(formula, data)

        # Get survey design
        survey_design = self.survey_design_spec.get_survey_design(
            regression_variable, X.index
        )

        # Create and fit the model
        model = SurveyModel(
            design=survey_design,
            model_class=sm.GLM,
            cov_method=self.cov_method,
            init_args=dict(family=self.family),
            fit_args=dict(use_t=self.use_t),
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
                result["Variable_pvalue"] = 1.0
                result["pvalue"] = 1.0
            else:
                result["Variable_pvalue"] = scipy.stats.t.sf(tval, df=dof) * 2
                result["pvalue"] = result["Variable_pvalue"]

        return result

    def _run_categorical_weighted(
        self, data, regression_variable, formula, formula_restricted
    ) -> Dict:
        """
        See:
        Lumley, Thomas, and Alastair Scott. "Tests for regression models fitted to survey data."
        Australian & New Zealand Journal of Statistics 56.1 (2014): 1-14.
        """
        result = dict()

        # Regress full model
        y, X = self._process_formula(formula, data)
        # Get survey design
        survey_design = self.survey_design_spec.get_survey_design(
            regression_variable, X.index
        )
        model = SurveyModel(
            design=survey_design,
            model_class=sm.GLM,
            cov_method=self.cov_method,
            init_args=dict(family=self.family),
            fit_args=dict(use_t=self.use_t),
        )
        model.fit(y=y, X=X)

        # Regress restricted model
        y_restricted, X_restricted = self._process_formula(formula_restricted, data)
        model_restricted = SurveyModel(
            design=survey_design,
            model_class=sm.GLM,
            cov_method=self.cov_method,
            init_args=dict(family=self.family),
            fit_args=dict(use_t=self.use_t),
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
            result["LRT_pvalue"] = lr_pvalue
            result["pvalue"] = result["LRT_pvalue"]
            # Don't report AIC values for weighted categorical analysis since they may be incorrect

        return result

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
                # Initialize result with placeholders
                result = self.get_default_result_dict(rv)
                result["Variable_type"] = rv_type
                # Run in a try/except block to catch any errors specific to a regression variable
                try:
                    # Take a copy of the data (ignoring other RVs) and create a keep_rows mask
                    keep_columns = [rv, self.outcome_variable] + self.covariates
                    data = self.data[keep_columns]

                    # Get missing weight mask
                    (
                        weight_name,
                        missing_weight_mask,
                        warning,
                    ) = self.survey_design_spec.check_missing_weights(data, rv)
                    if warning is not None:
                        self.warnings[rv].append(warning)
                    result["Weight"] = weight_name

                    # Get complete case mask
                    complete_case_mask = (
                        ~data[[rv, self.outcome_variable] + self.covariates]
                        .isna()
                        .any(axis=1)
                    )
                    # If allowed (an error hasn't been raised) negate missing_weight_mask so True=keep to drop those
                    complete_case_mask = complete_case_mask & ~missing_weight_mask

                    # Count restricted rows
                    restricted_rows = (
                        self.survey_design_spec.subset_array & complete_case_mask
                    )

                    # Filter by min_n
                    N = (restricted_rows).sum()
                    result["N"] = N
                    if N < self.min_n:
                        raise ValueError(
                            f"too few complete observations (min_n filter: {N} < {self.min_n})"
                        )

                    # Check for covariates that do not vary (they get ignored)
                    varying_covars, warnings = self._check_covariate_values(
                        restricted_rows
                    )
                    self.warnings[rv].extend(warnings)
                    data = data[
                        [rv, self.outcome_variable] + varying_covars
                    ]  # Drop any nonvarying covars

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
                            if self.survey_design_spec.subset_count > 0:
                                warning += f" and applying the {self.survey_design_spec.subset_count} subset(s)"
                            warning += f": {', '.join([repr(c) for c in extra_cats])} "
                            self.warnings[rv].append(warning)

                    # Get the formulas
                    formula_restricted, formula = self._get_formulas(rv, varying_covars)

                    # Run Regression
                    if rv_type == "continuous":
                        result.update(self._run_continuous_weighted(data, rv, formula))
                    elif (
                        rv_type == "binary"
                    ):  # The same calculation as for continuous variables
                        result.update(self._run_continuous_weighted(data, rv, formula))
                    elif rv_type == "categorical":
                        result.update(
                            self._run_categorical_weighted(
                                data, rv, formula, formula_restricted
                            )
                        )

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
