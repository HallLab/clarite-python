from typing import Optional, List

import click
import numpy as np
import pandas as pd
import patsy
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf

from ..modules.survey import SurveyDesignSpec, SurveyModel
from .calculations import regTermTest


class Regression(object):
    """
    """
    def __init__(self,
                 variable: str,
                 variable_kind: str,
                 phenotype: str,
                 phenotype_kind: str,
                 data: pd.DataFrame,
                 covariates: Optional[List] = None,
                 survey_design_spec: Optional[SurveyDesignSpec] = None,
                 cov_method: Optional[str] = 'error'):
        # Process input params
        self.variable = variable
        self.variable_kind = variable_kind
        self.phenotype = phenotype
        self.phenotype_kind = phenotype_kind
        self.data = data
        self.covariates = covariates
        self.cov_method = cov_method

        # Subset the data by dropping incomplete cases
        self.data = self.data.dropna(axis='index', how='any', subset=[self.variable, self.phenotype] + self.covariates)

        # Handle survey design
        self.survey_design = None
        if survey_design_spec is not None:
            # Get a survey design object based on the data
            survey_design, index = survey_design_spec.get_survey_design(self.variable, self.data.index)
            # Subset data based on the weights in the design
            self.data = self.data.loc[index]
            # Save the survey design
            self.survey_design = survey_design

        # Select regression family
        if phenotype_kind == "continuous":
            self.family = sm.families.Gaussian(link=sm.families.links.identity())
        elif phenotype_kind == 'binary':
            self.family = sm.families.Binomial(link=sm.families.links.logit())
        else:
            # TODO
            # Note: DoF might change
            raise NotImplementedError("Only continuous and binary phenotypes are currently supported")

        # Set default result values
        self.converged = False
        self.N = len(self.data)
        self.beta = np.nan
        self.SE = np.nan
        self.var_pvalue = np.nan
        self.LRT_pvalue = np.nan
        self.diff_AIC = np.nan
        self.pvalue = np.nan

    def check_covars(self):
        # No varying covariates if there aren't any
        if len(self.covariates) == 0:
            return []
        unique_values = self.data[self.covariates].nunique()
        varying_covars = list(unique_values[unique_values > 1].index.values)
        non_varying_covars = list(unique_values[unique_values <= 1].index.values)

        if len(non_varying_covars) > 0:
            click.echo(click.style(f"WARNING: {self.variable} has non-varying covariates(s): {', '.join(non_varying_covars)}", fg='yellow'))
        return varying_covars

    def run(self, min_n):
        """Run the regression and update self with the results"""
        # Check for a minimum amount of data
        if len(self.data) < min_n:
            click.echo(f"{self.variable} = NULL due to: too few complete obervations ({len(self.data)} < {min_n})")
            return
        self.varying_covariates = self.check_covars()

        # Make formulas
        self.formula_restricted = f"{self.phenotype} ~ "
        self.formula_restricted += " + ".join([f"C({var_name})" if str(self.data.dtypes[var_name]) == 'category' else var_name
                                               for var_name in self.varying_covariates])
        if str(self.data.dtypes[self.variable]) == 'category':
            self.formula = self.formula_restricted + f" + C({self.variable})"
        else:
            self.formula = self.formula_restricted + f" + {self.variable}"

        # Run Regression
        if self.survey_design is None:
            if self.variable_kind == 'continuous':
                self.run_continuous()
            elif self.variable_kind == 'binary':
                self.run_binary()  # Essentially same as continuous, except for the string used to key the results
            elif self.variable_kind == "categorical":
                self.run_categorical()
            else:
                raise ValueError(f"Unknown regression variable type '{self.variable_kind}'")
        else:
            if self.variable_kind == 'continuous':
                self.run_continuous_weighted()
            elif self.variable_kind == 'binary':
                self.run_binary_weighted()  # Same as continuous, at least for now
            elif self.variable_kind == "categorical":
                self.run_categorical_weighted()
            else:
                raise ValueError(f"Unknown regression variable type '{self.variable_kind}'")

    def get_results(self):
        """Return a dictionary of the results"""
        return {
            'Variable': self.variable,
            'Variable_type': self.variable_kind,
            'Converged': self.converged,
            'N': self.N,
            'Beta': self.beta,
            'SE': self.SE,
            'Variable_pvalue': self.var_pvalue,
            'LRT_pvalue': self.LRT_pvalue,
            'Diff_AIC': self.diff_AIC,
            'pvalue': self.pvalue
        }

    def run_continuous(self):
        # Regress
        est = smf.glm(self.formula, data=self.data, family=self.family).fit(use_t=True)
        # Check convergence
        if not est.converged:
            return
        else:
            self.converged = True
        # Get results
        self.beta = est.params[self.variable]
        self.SE = est.bse[self.variable]
        self.var_pvalue = est.pvalues[self.variable]
        self.pvalue = self.var_pvalue

    def run_binary(self):
        # Regress
        est = smf.glm(self.formula, data=self.data, family=self.family).fit(use_t=True)
        # Check convergence
        if not est.converged:
            return
        else:
            self.converged = True
        # Get results
        # Categorical-type RVs get a different name in the results, and aren't always at the end (since categorical come before non-categorical)
        rv_keys = [k for k in est.params.keys() if self.variable in k]
        try:
            assert len(rv_keys) == 1
            rv_key = rv_keys[0]
        except AssertionError:
            raise KeyError(f"Error extracting results for '{self.variable}', try renaming the variable")
        self.beta = est.params[rv_key]
        self.SE = est.bse[rv_key]
        self.var_pvalue = est.pvalues[rv_key]
        self.pvalue = self.var_pvalue

    def run_categorical(self):
        # Regress both models
        est_restricted = smf.glm(self.formula_restricted, data=self.data, family=self.family).fit(use_t=True)
        est = smf.glm(self.formula, data=self.data, family=self.family).fit(use_t=True)
        # Check convergence
        if not est.converged & est_restricted.converged:
            return
        else:
            self.converged = True
        # Calculate Results
        lrdf = (est_restricted.df_resid - est.df_resid)
        lrstat = -2*(est_restricted.llf - est.llf)
        lr_pvalue = scipy.stats.chi2.sf(lrstat, lrdf)
        self.LRT_pvalue = lr_pvalue
        self.pvalue = self.LRT_pvalue
        self.diff_AIC = est.aic - est_restricted.aic

    def run_continuous_weighted(self):
        y, X = patsy.dmatrices(self.formula, self.data, return_type='dataframe')
        # Create and fit the model
        model = SurveyModel(design=self.survey_design, model_class=sm.GLM, cov_method=self.cov_method,
                            init_args=dict(family=self.family),
                            fit_args=dict(use_t=True))
        model.fit(y=y, X=X)
        # Check convergence
        if not model.result.converged:
            return
        else:
            self.converged = True
        # Get results
        rv_idx_list = [i for i, n in enumerate(X.columns) if self.variable in n]
        if len(rv_idx_list) != 1:
            raise ValueError(f"Failed to find regression variable column in the results for {self.variable}")
        else:
            rv_idx = rv_idx_list[0]
        self.beta = model.params[rv_idx]
        self.SE = model.stderr[rv_idx]
        tval = np.abs(self.beta / self.SE)  # T statistic is the absolute value of beta / SE
        # Get degrees of freedom
        dof = self.survey_design.get_dof(X)
        self.var_pvalue = scipy.stats.t.sf(tval, df=dof)*2  # Two-sided t-test
        self.pvalue = self.var_pvalue

    def run_binary_weighted(self):
        y, X = patsy.dmatrices(self.formula, self.data, return_type='dataframe')
        # Create and fit the model
        model = SurveyModel(design=self.survey_design, model_class=sm.GLM, cov_method=self.cov_method,
                            init_args=dict(family=self.family),
                            fit_args=dict(use_t=True))
        model.fit(y=y, X=X)
        # Check convergence
        if not model.result.converged:
            return
        else:
            self.converged = True
        # Get results
        rv_idx_list = [i for i, n in enumerate(X.columns) if self.variable in n]
        if len(rv_idx_list) != 1:
            raise ValueError(f"Failed to find regression variable column in the results for {self.variable}")
        else:
            rv_idx = rv_idx_list[0]
        self.beta = model.params[rv_idx]
        self.SE = model.stderr[rv_idx]
        tval = np.abs(self.beta / self.SE)  # T statistic is the absolute value of beta / SE
        # Get degrees of freedom
        dof = self.survey_design.get_dof(X)
        self.var_pvalue = scipy.stats.t.sf(tval, df=dof)*2  # Two-sided t-test
        self.pvalue = self.var_pvalue

    def run_categorical_weighted(self):
        # The change in deviance between a model and a nested version (with n fewer predictors) follows a chi-square distribution with n DoF
        # See https://en.wikipedia.org/wiki/Deviance_(statistics)
        # Regress restricted model
        y, X_restricted = patsy.dmatrices(self.formula_restricted, self.data, return_type='dataframe')
        model_restricted = SurveyModel(design=self.survey_design, model_class=sm.GLM, cov_method=self.cov_method,
                                       init_args=dict(family=self.family),
                                       fit_args=dict(use_t=True))
        model_restricted.fit(y=y, X=X_restricted)
        # Regress full model (Already have the survey_design and index objects)
        y, X = patsy.dmatrices(self.formula, self.data, return_type='dataframe')
        model = SurveyModel(design=self.survey_design, model_class=sm.GLM, cov_method=self.cov_method,
                            init_args=dict(family=self.family),
                            fit_args=dict(use_t=True))
        model.fit(y=y, X=X)
        # Check convergence
        if not model.result.converged & model_restricted.result.converged:
            return
        else:
            self.converged = True
        # Calculate Results
        dof = self.survey_design.get_dof(X)
        lr_pvalue = regTermTest(full_model=model, restricted_model=model_restricted, ddf=dof, X_names=X.columns, var_name=self.variable)
        # Gather Other Results
        self.LRT_pvalue = lr_pvalue
        self.pvalue = self.LRT_pvalue
        # Don't report AIC values for weighted categorical analysis since they may be incorrect
        # self.diff_AIC = model.result.aic - model_restricted.result.aic
        self.diff_AIC = np.nan
