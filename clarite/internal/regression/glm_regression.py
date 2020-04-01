from typing import Dict, Optional, List, Tuple

import numpy as np
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf

from clarite.internal.utilities import _remove_empty_categories, _get_dtype

from .base import Regression


class GLMRegression(Regression):
    """
    Statsmodels GLM Regression.
    """
    def __init__(self, data, outcome_variable, test_variable, covariates, min_n):
        """
        Parameters
        ----------
        data - pd.DataFrame
        outcome_variable - name of the outcome variable
        test_variable - name of the variable being tested
        covariates - other variables to include in the regression formula
        min_n - minimum number of observations (after discarding any with NA)
        """
        # base class init (sets pvalue, test_dtype, error, and warnings
        super().__init__()

        # Store passed parameters
        self.data = data
        self.outcome_variable = outcome_variable
        self.outcome_dtype = _get_dtype(data[outcome_variable])
        self.test_variable = test_variable
        self.test_dtype = _get_dtype(data[test_variable])
        self.covariates = covariates
        self.min_n = min_n

        # Placeholders
        self.varying_covars = []
        self.non_varying_covars = []
        self.formula_restricted = ""
        self.formula = ""
        self.family = None

        # Set default result values
        self.converged = False
        self.N = len(self.data)
        self.beta = np.nan
        self.SE = np.nan
        self.var_pvalue = np.nan
        self.LRT_pvalue = np.nan
        self.diff_AIC = np.nan

    def subset_data(self):
        """Remove observations with missing data and run the min_n filter"""
        self.data = self.data.dropna(axis='index', how='any',
                                     subset=[self.test_variable, self.outcome_variable] + self.covariates)

    def check_covariate_values(self):
        """Remove covariates that do not vary"""
        unique_values = self.data[self.covariates].nunique()
        self.varying_covars = list(unique_values[unique_values > 1].index.values)
        self.non_varying_covars = list(unique_values[unique_values <= 1].index.values)
        if len(self.non_varying_covars) > 0:
            self.warnings.append(f"non-varying covariates(s): {', '.join(self.non_varying_covars)}")

    def remove_extra_categories(self):
        """Check for extra categories after filtering and warn"""
        removed_cats = _remove_empty_categories(self.data, only=[self.test_variable, ] + self.varying_covars)
        if len(removed_cats) >= 1:
            for extra_cat_var, extra_cats in removed_cats.items():
                self.warnings.append(f"'{str(extra_cat_var)}' had categories with no occurrences: "
                                     f"{', '.join([str(c) for c in extra_cats])} "
                                     f"after removing observations with missing values")

    def get_formulas(self):
        # Restricted Formula
        self.formula_restricted = f"{self.outcome_variable} ~ "
        self.formula_restricted += " + ".join(
            [f"C({var_name})"
             if str(self.data.dtypes[var_name]) == 'category'
             else var_name for var_name in self.varying_covars])

        # Full Formula
        if str(self.data.dtypes[self.test_variable]) == 'category':
            self.formula = self.formula_restricted + f" + C({self.test_variable})"
        else:
            self.formula = self.formula_restricted + f" + {self.test_variable}"

    def pre_run_setup(self):
        # Subset Data
        self.subset_data()

        # Minimum complete cases filter
        if len(self.data) < self.min_n:
            self.error = f"{self.test_variable} = NULL due to: too few complete observations ({len(self.data)} < {self.min_n})"
            return

        # Check variable values, creating warnings if needed
        self.check_covariate_values()
        self.remove_extra_categories()

        # Get the formulas that will be used
        self.get_formulas()

        # Select regression family
        if self.outcome_dtype == "continuous":
            self.family = sm.families.Gaussian(link=sm.families.links.identity())
        elif self.outcome_dtype == 'binary':
            self.family = sm.families.Binomial(link=sm.families.links.logit())
        else:
            raise NotImplementedError("Only continuous and binary phenotypes are currently supported for GLMRegression")

    def get_result(self) -> Tuple[Dict, List[str], Optional[str]]:
        """
        Return results as a dict along with any warnings and error
        """
        result = {
            'Variable': self.test_variable,
            'Variable_type': self.test_dtype,
            'Converged': self.converged,
            'N': self.N,
            'Beta': self.beta,
            'SE': self.SE,
            'Variable_pvalue': self.var_pvalue,
            'LRT_pvalue': self.LRT_pvalue,
            'Diff_AIC': self.diff_AIC,
            'pvalue': self.pvalue
        }
        return result, self.warnings, self.error

    def run_continuous(self):
        # Regress
        est = smf.glm(self.formula, data=self.data, family=self.family).fit(use_t=True)
        # Check convergence
        if not est.converged:
            return
        else:
            self.converged = True
        # Get results
        self.beta = est.params[self.test_variable]
        self.SE = est.bse[self.test_variable]
        self.var_pvalue = est.pvalues[self.test_variable]
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
        # Categorical-type RVs get a different name in the results, and aren't always at the end
        # (since categorical come before non-categorical)
        rv_keys = [k for k in est.params.keys() if self.test_variable in k]
        try:
            assert len(rv_keys) == 1
            rv_key = rv_keys[0]
        except AssertionError:
            self.error = f"Error extracting results for '{self.test_variable}', try renaming the variable"
            return
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
