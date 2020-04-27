from typing import Optional

import numpy as np
import scipy
import patsy
import statsmodels.api as sm

from .glm_regression import GLMRegression
from clarite.modules.survey import SurveyDesignSpec, SurveyModel
from clarite.internal.calculations import regTermTest


class WeightedGLMRegression(GLMRegression):
    """Overwrite a few methods of the parent GLMRegression class to use survey weights"""
    def __init__(self, data, outcome_variable, outcome_dtype, test_variable, covariates, min_n,
                 survey_design_spec: SurveyDesignSpec, cov_method: Optional[str] = 'error'):
        super().__init__(data, outcome_variable, outcome_dtype, test_variable, covariates, min_n)
        self.survey_design_spec = survey_design_spec
        self.cov_method = cov_method
        self.survey_design = None

    def pre_run_setup(self):
        # Run the original pre-run setup
        super().pre_run_setup()
        # Get a survey design object based on the non-missing data
        self.survey_design = self.survey_design_spec.get_survey_design(self.test_variable, self.complete_case_idx)
        # Raise an error if the survey design is missing weights when the variable value is not
        variable_na = self.data[self.test_variable].isna()
        weight_na = self.survey_design.weights.isna()
        values_with_missing = self.data.loc[~variable_na & weight_na, self.test_variable]
        # Get unique values
        unique_missing = values_with_missing.unique()
        unique_not_missing = self.data.loc[~variable_na & ~weight_na, self.test_variable].unique()
        sometimes_missing = sorted([str(v) for v in (set(unique_missing) & set(unique_not_missing))])
        always_missing = sorted([str(v) for v in (set(unique_missing) - set(unique_not_missing))])
        # Log
        if len(values_with_missing) > 0:
            error = f"{len(values_with_missing):,} observations are missing weights when the variable is not missing."
            # Log sometimes missing values
            if len(sometimes_missing) == 0:
                pass
            elif len(sometimes_missing) == 1:
                error += f"\n\tOne value has a missing weight sometimes: {sometimes_missing[0]}"
            elif len(sometimes_missing) <= 5:
                error += f"\n\t{len(sometimes_missing)} values have a missing weight sometimes: {', '.join(sometimes_missing)}"
            elif len(sometimes_missing) > 5:
                error += f"\n\t{len(sometimes_missing)} values have a missing weight sometimes: {', '.join(sometimes_missing[:5])}, ..."
            # Log always missing values
            if len(always_missing) == 0:
                pass
            elif len(always_missing) == 1:
                error += f"\n\tOne value is always missing weights: {always_missing[0]}. Should it be encoded as NaN?"
            elif len(always_missing) <= 5:
                error += f"\n\t{len(always_missing)} values are always missing weights: {', '.join(always_missing)}." \
                         f" Should they be encoded as NaN?"
            elif len(always_missing) > 5:
                error += f"\n\t{len(always_missing)} values are always missing weights: {', '.join(always_missing[:5])}, ..." \
                         f" Should they be encoded as NaN?"
            # Raise the error
            raise ValueError(error)

    def run_continuous(self):
        y, X = patsy.dmatrices(self.formula, self.data, return_type='dataframe', NA_action='drop')
        # Create and fit the model
        model = SurveyModel(design=self.survey_design, model_class=sm.GLM, cov_method=self.cov_method,
                            init_args=dict(family=self.family),
                            fit_args=dict(use_t=self.use_t))
        model.fit(y=y, X=X)
        # Check convergence
        if not model.result.converged:
            return
        else:
            self.converged = True
        # Get results
        rv_idx_list = [i for i, n in enumerate(X.columns) if self.test_variable in n]
        if len(rv_idx_list) != 1:
            raise ValueError(f"Failed to find regression variable column in the results for {self.test_variable}")
        else:
            rv_idx = rv_idx_list[0]
        self.beta = model.params[rv_idx]
        self.SE = model.stderr[rv_idx]
        tval = np.abs(self.beta / self.SE)  # T statistic is the absolute value of beta / SE
        # Get degrees of freedom
        dof = self.survey_design.get_dof(X)
        self.var_pvalue = scipy.stats.t.sf(tval, df=dof)*2  # Two-sided t-test
        self.pvalue = self.var_pvalue

    def run_binary(self):
        y, X = patsy.dmatrices(self.formula, self.data, return_type='dataframe', NA_action='drop')
        # Create and fit the model
        model = SurveyModel(design=self.survey_design, model_class=sm.GLM, cov_method=self.cov_method,
                            init_args=dict(family=self.family),
                            fit_args=dict(use_t=self.use_t))
        model.fit(y=y, X=X)
        # Check convergence
        if not model.result.converged:
            return
        else:
            self.converged = True
        # Get results
        rv_idx_list = [i for i, n in enumerate(X.columns) if self.test_variable in n]
        if len(rv_idx_list) != 1:
            raise ValueError(f"Failed to find regression variable column in the results for {self.test_variable}")
        else:
            rv_idx = rv_idx_list[0]
        self.beta = model.params[rv_idx]
        self.SE = model.stderr[rv_idx]
        tval = np.abs(self.beta / self.SE)  # T statistic is the absolute value of beta / SE
        # Get degrees of freedom
        dof = self.survey_design.get_dof(X)
        self.var_pvalue = scipy.stats.t.sf(tval, df=dof)*2  # Two-sided t-test
        self.pvalue = self.var_pvalue

    def run_categorical(self):
        # The change in deviance between a model and a nested version (with n fewer predictors) follows a chi-square distribution with n DoF
        # See https://en.wikipedia.org/wiki/Deviance_(statistics)
        # Regress full model
        y, X = patsy.dmatrices(self.formula, self.data, return_type='dataframe', NA_action='drop')
        model = SurveyModel(design=self.survey_design, model_class=sm.GLM, cov_method=self.cov_method,
                            init_args=dict(family=self.family),
                            fit_args=dict(use_t=self.use_t))
        model.fit(y=y, X=X)
        # Regress restricted model
        # Use same X and y (but fewer columns in X) to ensure correct comparison
        _, X_restricted = patsy.dmatrices(self.formula_restricted, self.data, return_type='dataframe', NA_action='drop')
        X_restricted = X_restricted.loc[X.index]
        model_restricted = SurveyModel(design=self.survey_design, model_class=sm.GLM, cov_method=self.cov_method,
                                       init_args=dict(family=self.family),
                                       fit_args=dict(use_t=self.use_t))
        model_restricted.fit(y=y, X=X_restricted)
        # Check convergence
        if not model.result.converged & model_restricted.result.converged:
            return
        else:
            self.converged = True
        # Calculate Results
        dof = self.survey_design.get_dof(X)
        lr_pvalue = regTermTest(full_model=model, restricted_model=model_restricted,
                                ddf=dof, X_names=X.columns, var_name=self.test_variable)
        # Gather Other Results
        self.LRT_pvalue = lr_pvalue
        self.pvalue = self.LRT_pvalue
        # Don't report AIC values for weighted categorical analysis since they may be incorrect
        # self.diff_AIC = model.result.aic - model_restricted.result.aic
        self.diff_AIC = np.nan
