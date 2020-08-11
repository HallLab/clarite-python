from typing import Optional, Dict, Tuple, List

import click
import numpy as np
import scipy
import patsy
import statsmodels.api as sm

from .glm_regression import GLMRegression
from clarite.modules.survey import SurveyDesignSpec, SurveyModel
from clarite.internal.calculations import regTermTest
from ..utilities import _remove_empty_categories


class WeightedGLMRegression(GLMRegression):
    """
    Statsmodels GLM Regression with adjustments for survey design

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
        Minimum number of complete-case observations (no NA values for phenotype, covariates, variable, or weight)
        Defaults to 200
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
    >>> ewas_discovery = clarite.analyze.ewas("logBMI", covariates, nhanes_discovery)
    Running EWAS on a continuous variable
    """
    def __init__(self, data, outcome_variable, covariates,
                 survey_design_spec: SurveyDesignSpec,
                 min_n: int = 200,
                 cov_method: Optional[str] = 'stata'):
        super().__init__(data=data,
                         outcome_variable=outcome_variable,
                         covariates=covariates,
                         min_n=min_n)

        # Custom init involving kwargs passed to this regression
        self.survey_design_spec = survey_design_spec
        self.cov_method = cov_method

        # Add survey design info to the description
        self.description += "\n" + str(self.survey_design_spec)

    @staticmethod
    def get_default_result_dict():
        return {'Weight': "",
                'Converged': False,
                'N': np.nan,
                'Beta': np.nan,
                'SE': np.nan,
                'Variable_pvalue': np.nan,
                'LRT_pvalue': np.nan,
                'Diff_AIC': np.nan,
                'pvalue': np.nan}

    def run_continuous_weighted(self, data, survey_design, regression_variable, complete_case_idx, formula) -> Dict:
        result = dict()
        # Get data based on the formula
        y, X = patsy.dmatrices(formula, data, return_type='dataframe', NA_action='drop')

        # Create and fit the model
        model = SurveyModel(design=survey_design, model_class=sm.GLM, cov_method=self.cov_method,
                            init_args=dict(family=self.family),
                            fit_args=dict(use_t=self.use_t))
        model.fit(y=y, X=X)

        # Save results if the regression converged
        if model.result.converged:
            result['Converged'] = True
            rv_idx_list = [i for i, n in enumerate(X.columns) if regression_variable in n]
            if len(rv_idx_list) != 1:
                raise ValueError(f"Failed to find regression variable column in the results for {regression_variable}")
            else:
                rv_idx = rv_idx_list[0]
            result['Beta'] = model.params[rv_idx]
            result['SE'] = model.stderr[rv_idx]
            # Calculate pvalue using a Two-sided t-test
            tval = np.abs(result['Beta'] / result['SE'])  # T statistic is the absolute value of beta / SE
            dof = survey_design.get_dof(X)
            result['Variable_pvalue'] = scipy.stats.t.sf(tval, df=dof)*2
            result['pvalue'] = result['Variable_pvalue']

        return result

    def run_categorical_weighted(self, data, survey_design, regression_variable, complete_case_idx,
                                 formula, formula_restricted) -> Dict:
        """
        The change in deviance between a model and a nested version (with n fewer predictors) follows a chi-square distribution with n DoF
        See https://en.wikipedia.org/wiki/Deviance_(statistics)
        """
        result = dict()

        # Regress full model
        y, X = patsy.dmatrices(formula, data, return_type='dataframe', NA_action='drop')
        model = SurveyModel(design=survey_design, model_class=sm.GLM, cov_method=self.cov_method,
                            init_args=dict(family=self.family),
                            fit_args=dict(use_t=self.use_t))
        model.fit(y=y, X=X)

        # Regress restricted model
        # Use same X and y (but fewer columns in X) to ensure correct comparison
        _, X_restricted = patsy.dmatrices(formula_restricted, data, return_type='dataframe', NA_action='drop')
        X_restricted = X_restricted.loc[X.index]
        model_restricted = SurveyModel(design=survey_design, model_class=sm.GLM, cov_method=self.cov_method,
                                       init_args=dict(family=self.family),
                                       fit_args=dict(use_t=self.use_t))
        model_restricted.fit(y=y, X=X_restricted)

        # Save results if the regression converged
        if model.result.converged & model_restricted.result.converged:
            result['Converged'] = True
            dof = survey_design.get_dof(X)
            lr_pvalue = regTermTest(full_model=model, restricted_model=model_restricted,
                                    ddf=dof, X_names=X.columns, var_name=regression_variable)
            result['LRT_pvalue'] = lr_pvalue
            result['pvalue'] = result['LRT_pvalue']
            # Don't report AIC values for weighted categorical analysis since they may be incorrect

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
                    # Take a copy of the required variables rather than operating directly on the stored data
                    columns = [rv, self.outcome_variable] + self.covariates
                    data = self.data.loc[self.survey_design_spec.subset_array, columns]
                    _remove_empty_categories(data)  # Make sure later warning is just about after filtering

                    # Check for missing weights, potentially dropping them
                    _, weight_name, _ = self.survey_design_spec.get_weights(rv)
                    missing_weight_idx, warning = self.survey_design_spec.check_missing_weights(data, rv)
                    if missing_weight_idx is not None:
                        # Drop those rows
                        data = data.drop(missing_weight_idx, axis=0)
                        # Update weight name
                        weight_name += f" ({len(missing_weight_idx)} observations are missing weights)"
                        # Save warning
                        self.warnings[rv].append(warning)

                    # Get survey design for the given variable
                    complete_case_idx = self.get_complete_case_idx(data, rv)
                    survey_design, data = self.survey_design_spec.get_survey_design(data, rv, complete_case_idx)

                    # Get updated complete case index and filter by min_n
                    complete_case_idx = self.get_complete_case_idx(data, rv)
                    N = len(complete_case_idx)
                    self.results[rv]['N'] = N
                    if N < self.min_n:
                        raise ValueError(f"too few complete observations (min_n filter: {N} < {self.min_n})")

                    # Check for covariates that do not vary (they get ignored)
                    varying_covars, warnings = self.check_covariate_values(complete_case_idx)
                    self.warnings[rv].extend(warnings)
                    data = data[[rv, self.outcome_variable] + varying_covars]  # Drop any nonvarying covars

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
                        result = self.run_continuous_weighted(data, survey_design, rv, complete_case_idx, formula)
                    elif rv_type == 'binary':  # The same calculation as for continuous variables
                        result = self.run_continuous_weighted(data, survey_design, rv, complete_case_idx, formula)
                    elif rv_type == 'categorical':
                        result = self.run_categorical_weighted(data, survey_design, rv, complete_case_idx,
                                                               formula, formula_restricted)
                    else:
                        result = dict()
                    self.results[rv].update(result)

                except Exception as e:
                    self.errors[rv] = str(e)

            click.echo(click.style(f"\tFinished Running {len(rv_list):,} {rv_type} variables", fg='green'))
        self.run_complete = True
