import re
from itertools import combinations
from typing import Dict, List, Tuple

import click
import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf

from clarite.internal.utilities import _remove_empty_categories
from . import GLMRegression

from .base import Regression


class InteractionRegression(GLMRegression):
    """
    Statsmodels GLM Regression.
    This class handles running regressions and calculating LRT pvalues based on including interaction terms

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
    interactions: list(tuple(strings)), str, or None
        Valid variables are those in the data that are not the outcome variable or a covariate.
        None: Test all pairwise interactions between valid variables
        String: Test all interactions of this valid variable with other valid variables
        List of tuples: Test specific interactions of valid variables
    report_betas: boolean
        False by default.
          If True, the results will contain one row for each interaction term and will include the beta value
          for that term.  The number of terms increases with the number of categories in each interacting term.

    """

    def __init__(
        self,
        data,
        outcome_variable,
        covariates,
        min_n=200,
        interactions=None,
        report_betas=False,
    ):
        # base class init
        # This takes in minimal regression params (data, outcome_variable, covariates) and
        # initializes additional parameters (outcome dtype, regression variables, error, and warnings)
        super().__init__(
            data=data,
            outcome_variable=outcome_variable,
            covariates=covariates,
            min_n=min_n,
        )

        # Custom init involving kwargs passed to this regression
        self.report_betas = report_betas
        self._process_interactions(interactions)

        # Use a list of results instead of the default dict
        self.results = []

    def _process_interactions(self, interactions):
        """Validate the interactions parameter and save it as a list of string tuples"""
        regression_var_list = (
            self.regression_variables["binary"]
            + self.regression_variables["categorical"]
            + self.regression_variables["continuous"]
        )
        if len(regression_var_list) < 2:
            raise ValueError(
                f"Not enough valid variables for running interactions: {len(regression_var_list)} variables"
            )
        if interactions is None:
            self.interactions = [c for c in combinations(regression_var_list, r=2)]
        elif type(interactions) == str:
            if interactions not in regression_var_list:
                raise ValueError(
                    f"'{interactions}' was passed as the value for 'interactions' "
                    f"but is not a valid variable"
                )
        else:
            # Check all interactions include two variables that are present
            errors = []
            for idx, i in enumerate(interactions):
                if len(i) != 2:
                    errors.append(
                        f"Interaction {idx + 1} of {len(interactions)} does not list exactly two variables."
                    )
                elif i[0] not in regression_var_list:
                    errors.append(
                        f"Interaction {idx + 1} of {len(interactions)} contains an invalid variable: '{i[0]}'"
                    )
                elif i[1] not in regression_var_list:
                    errors.append(
                        f"Interaction {idx + 1} of {len(interactions)} contains an invalid variable: '{i[1]}'"
                    )
            if len(errors) > 0:
                raise ValueError(
                    "Invalid interactions provided\n" + "\n\t".join(errors)
                )
            else:
                self.interactions = interactions
        self.description += f"\nProcessing {len(self.interactions):,} interactions"

    @staticmethod
    def _get_default_result_dict(i1, i2):
        return {
            "Term1": i1,
            "Term2": i2,
            "Converged": False,
            "N": np.nan,
            "Beta": np.nan,
            "SE": np.nan,
            "Beta_pvalue": np.nan,
            "LRT_pvalue": np.nan,
        }

    def _get_formulas(self, i1, i2, varying_covars) -> Tuple[str, str]:
        # Restricted Formula - covariates and main effects
        formula_restricted = f"{self.outcome_variable} ~ 1 + {i1} + {i2}"
        if len(varying_covars) > 0:
            formula_restricted += " + "
            formula_restricted += " + ".join(varying_covars)

        # Full Formula - restricted plus interactions
        formula = formula_restricted + f" + {i1}:{i2}"

        return formula_restricted, formula

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
        result["Phenotype"] = self.outcome_variable
        if self.report_betas:
            return result.set_index(
                ["Term1", "Term2", "Phenotype", "Parameter"]
            ).sort_values(["LRT_pvalue", "Beta_pvalue"])
        else:
            return result.set_index(["Term1", "Term2", "Phenotype"]).sort_values(
                ["LRT_pvalue"]
            )

    def _run_interaction(self, data, formula, formula_restricted) -> Dict:
        # Regress both models
        est = smf.glm(formula, data=data, family=self.family).fit(use_t=self.use_t)
        est_restricted = smf.glm(formula_restricted, data=data, family=self.family).fit(
            use_t=True
        )
        # Check convergence
        if est.converged & est_restricted.converged:
            lrdf = est_restricted.df_resid - est.df_resid
            lrstat = -2 * (est_restricted.llf - est.llf)
            lr_pvalue = scipy.stats.chi2.sf(lrstat, lrdf)
            if self.report_betas:
                # Get beta, SE, and pvalue from interaction terms
                # Where interaction terms are those appearing in the full model and not in the reduced model
                # Return all terms
                param_names = set(est.bse.index) - set(est_restricted.bse.index)
                # The restricted model shouldn't have extra terms, unless there is some case we have overlooked
                assert len(set(est_restricted.bse.index) - set(est.bse.index)) == 0
                for param_name in param_names:
                    yield {
                        "Converged": True,
                        "Parameter": param_name,
                        "Beta": est.params[param_name],
                        "SE": est.bse[param_name],
                        "Beta_pvalue": est.pvalues[param_name],
                        "LRT_pvalue": lr_pvalue,
                    }
            else:
                # Only return the LRT result
                yield {"Converged": True, "LRT_pvalue": lr_pvalue}
        else:
            # Did not converge - nothing to update
            yield dict()

    def run(self):
        """Run a regression object, returning the results and logging any warnings/errors"""
        for idx, interaction in enumerate(self.interactions):
            i1, i2 = interaction
            interaction_num = idx + 1
            if interaction_num % 100 == 0:
                click.echo(
                    click.style(
                        f"Running {interaction_num:,} of {len(self.interactions):,} interactions...",
                        fg="green",
                    )
                )
            interaction_str = f"{i1}:{i2}"
            # Run in a try/except block to catch any errors specific to a regression variable
            try:
                # Take a copy of the data (ignoring other RVs)
                keep_columns = [i1, i2, self.outcome_variable] + self.covariates
                data = self.data[keep_columns]

                # Get complete case mask and filter by min_n
                complete_case_mask = (
                    ~data[[i1, i2, self.outcome_variable] + self.covariates]
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
                self.warnings[interaction_str].extend(warnings)

                # Remove unused categories (warning when this occurs)
                removed_cats = _remove_empty_categories(data)
                if len(removed_cats) >= 1:
                    for extra_cat_var, extra_cats in removed_cats.items():
                        self.warnings[interaction_str].append(
                            f"'{str(extra_cat_var)}'"
                            f" had categories with no occurrences: "
                            f"{', '.join([str(c) for c in extra_cats])} "
                            f"after removing observations with missing values"
                        )

                # Get the formulas
                formula_restricted, formula = self._get_formulas(i1, i2, varying_covars)

                # Apply the complete_case_mask to the data to ensure categorical models use the same data in the LRT
                data = data.loc[complete_case_mask]

                # Run Regression LRT Test
                for regression_result in self._run_interaction(
                    data, formula, formula_restricted
                ):
                    result = self._get_default_result_dict(i1, i2)
                    result["N"] = N
                    result.update(regression_result)
                    self.results.append(result)

            except Exception as e:
                self.errors[interaction_str] = str(e)
                result = self._get_default_result_dict(i1, i2)
                result["N"] = N
                self.results.append(result)

        self.run_complete = True
