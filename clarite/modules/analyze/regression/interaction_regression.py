import multiprocessing
from itertools import combinations, repeat
from typing import Dict, Generator, List, Optional, Tuple

import click
import numpy as np
import pandas as pd
import patsy
import scipy
import statsmodels.api as sm
from pandas_genomics import GenotypeDtype

from clarite.internal.utilities import _remove_empty_categories

from ..utils import fix_names
from . import GLMRegression

# GITHUB ISSUE #119: Regressions with Error after Multiprocessing release python > 3.8
# multiprocessing.get_start_method("fork")


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
        The data to be analyzed, including the outcome, covariates, and any variables to be regressed.
    outcome_variable: string
        The variable to be used as the output (y) of the regression
    covariates: list (strings),
        The variables to be used as covariates.  Any variables in the DataFrames not listed as covariates are regressed.
    min_n: int or None
        Minimum number of complete-case observations (no NA values for outcome, covariates, or variable)
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
    encoding: str, default "additive"
        Encoding method to use for any genotype data.  One of {'additive', 'dominant', 'recessive', 'codominant', or 'weighted'}
    edge_encoding_info: Optional pd.DataFrame, default None
        If edge encoding is used, this must be provided.  See Pandas-Genomics documentation on edge encodings.
    process_num: Optional[int]
        Number of processes to use when running the analysis, default is None (use the number of cores)

    """

    def __init__(
        self,
        data,
        outcome_variable,
        covariates,
        min_n=200,
        interactions=None,
        report_betas=False,
        encoding: str = "additive",
        edge_encoding_info: Optional[pd.DataFrame] = None,
        process_num: Optional[int] = None,
    ):
        # base class init
        # This takes in minimal regression params (data, outcome_variable, covariates) and
        # initializes additional parameters (outcome dtype, regression variables, error, and warnings)
        regression_variables = list(
            set(data.columns) - {outcome_variable} - set(covariates)
        )
        super().__init__(
            data=data,
            outcome_variable=outcome_variable,
            covariates=covariates,
            regression_variables=regression_variables,
            encoding=encoding,
            edge_encoding_info=edge_encoding_info,
            min_n=min_n,
        )

        # Custom init involving kwargs passed to this regression
        self.report_betas = report_betas
        self._process_interactions(interactions)
        if process_num is None:
            process_num = multiprocessing.cpu_count()
        self.process_num = process_num

        # Use a list of results instead of the default dict
        self.results = []

    def _process_interactions(self, interactions):
        """Validate the interactions parameter and save it as a list of string tuples"""
        regression_var_list = []
        for var_list in self.regression_variables.values():
            regression_var_list.extend(var_list)

        if len(regression_var_list) < 2:
            raise ValueError(
                f"Not enough valid variables for running interactions: {len(regression_var_list)} variables"
            )
        if interactions is None:
            self.interactions = [c for c in combinations(regression_var_list, r=2)]
        elif type(interactions) is str:
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
                    "Invalid interactions provided\n\t" + "\n\t".join(errors)
                )
            else:
                self.interactions = interactions
        self.description += f"\nProcessing {len(self.interactions):,} interactions"

    @staticmethod
    def _get_default_result_dict(i1, i2, outcome_variable):
        return {
            "Outcome": outcome_variable,
            "Term1": i1,
            "Term2": i2,
            "Parameter": str(i1 + ":" + i2),
            "Converged": False,
            "N": np.nan,
            "LRT_pvalue": np.nan,
            "Red_Var1_beta": np.nan,
            "Red_Var1_SE": np.nan,
            "Red_Var1_Pval": np.nan,
            "Red_Var2_beta": np.nan,
            "Red_Var2_SE": np.nan,
            "Red_Var2_Pval": np.nan,
            "Full_Var1_Var2_beta": np.nan,
            "Full_Var1_Var2_SE": np.nan,
            "Full_Var1_Var2_Pval": np.nan,
            "Full_Var1_beta": np.nan,
            "Full_Var1_SE": np.nan,
            "Full_Var1_Pval": np.nan,
            "Full_Var2_beta": np.nan,
            "Full_Var2_SE": np.nan,
            "Full_Var2_Pval": np.nan,
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
        result["Outcome"] = self.outcome_variable
        if self.report_betas:
            return result.set_index(
                # ["Term1", "Term2", "Outcome", "Parameter"]
                ["Term1", "Term2", "Outcome"]
            ).sort_values(["LRT_pvalue", "Full_Var1_Var2_Pval"])
        else:
            return result.set_index(["Term1", "Term2", "Outcome"]).sort_values(
                ["LRT_pvalue"]
            )

    @staticmethod
    def _run_interaction_regression(
        data, formula, formula_restricted, family, use_t, report_betas, i1, i2
    ) -> Generator[Dict, None, None]:
        # Regress Full Model
        y, X = patsy.dmatrices(formula, data, return_type="dataframe", NA_action="drop")
        y = fix_names(y)
        X = fix_names(X)
        est = sm.GLM(y, X, family=family).fit(use_t=use_t)

        # Regress Restricted Model
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
            lrdf = est_restricted.df_resid - est.df_resid
            lrstat = -2 * (est_restricted.llf - est.llf)
            lr_pvalue = scipy.stats.chi2.sf(lrstat, lrdf)
            # GITHUB/ISSUES 121: Handling LRT_Pvalue when lrstat and lrdf are
            # both 0. When lrstat (the test statistic) and lrdf (degrees of
            # freedom for the Likelihood Ratio Test) are both 0, it typically
            # suggests that both models are equivalent in terms of fit. In
            # other words, there is no significant difference between the two
            # models.
            #
            # However when both lrstat and lrdf are 0, calc the survival
            # function (sf) of a chi-squared distribution with 0 degrees of
            # freedom results in NaN. This is because mathematically, it's
            # undefined to perform this calculation under these circumstances.
            #
            # In such cases, it's important to handle this scenario separately
            # in the result based on the specific requirements of the analysis
            if lrdf == 0 and lrstat == 0:
                # Both models are equal
                yield {"Converged": False, "LRT_pvalue": lr_pvalue}
            if np.isnan(lr_pvalue):
                # There is an issue with the LRT calculation
                yield {"Converged": False, "LRT_pvalue": lr_pvalue}
            else:
                if report_betas:
                    # Get beta, SE, and pvalue from interaction terms
                    # Where interaction terms are those appearing in the full
                    # model and not in the reduced model return all terms
                    param_names = set(est.bse.index) - set(est_restricted.bse.index)
                    # The restricted model shouldn't have extra terms, unless
                    # there is some case we have overlooked.
                    assert len(set(est_restricted.bse.index) - set(est.bse.index)) == 0
                    # GITHUB/ISSUES 122: Open to show Terms Betas Values
                    for param_name in param_names:
                        # Names defined to aling with PLATO
                        # Split the input_string by ":"
                        term_1, term_2 = param_name.split(":")
                        yield {
                            "Term1": term_1,
                            "Term2": term_2,
                            "Converged": True,
                            "Parameter": param_name,
                            # Betas in Reduced Model
                            # Var1 --> Term 1
                            "Red_Var1_beta": est_restricted.params[term_1],
                            "Red_Var1_SE": est_restricted.bse[term_1],
                            "Red_Var1_Pval": est_restricted.pvalues[term_1],
                            # Var2 --> Term 2
                            "Red_Var2_beta": est_restricted.params[term_2],
                            "Red_Var2_SE": est_restricted.bse[term_2],
                            "Red_Var2_Pval": est_restricted.pvalues[term_2],
                            # Betas in Full Model
                            # Var1 --> Term 1
                            "Full_Var1_Var2_beta": est.params[param_name],
                            "Full_Var1_Var2_SE": est.bse[param_name],
                            "Full_Var1_Var2_Pval": est.pvalues[param_name],
                            # Var1 --> Term 1
                            "Full_Var1_beta": est.params[term_1],
                            "Full_Var1_SE": est.bse[term_1],
                            "Full_Var1_Pval": est.pvalues[term_1],
                            # Var2 --> Term 2
                            "Full_Var2_beta": est.params[term_2],
                            "Full_Var2_SE": est.bse[term_2],
                            "Full_Var2_Pval": est.pvalues[term_2],
                            "LRT_pvalue": lr_pvalue,
                        }
                else:
                    # Only return the LRT result
                    yield {"Converged": True, "LRT_pvalue": lr_pvalue}

        else:
            # Did not converge - nothing to update
            yield dict()

    def _get_interaction_specific_data(self, interaction: Tuple[str, str]):
        """Select the data relevant to performing a regression on a given interaction, encoding genotypes if needed"""
        data = self.data[
            list(interaction)
            + [
                self.outcome_variable,
            ]
            + self.covariates
        ].copy()

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
        # Log how many interactions are being run using how many processes
        click.echo(
            click.style(
                f"Running {len(self.interactions):,} interactions using {self.process_num} processes...",
                fg="green",
            )
        )

        # TODO: Error on multiprocess after update to Python > 3.8
        self.process_num = 1

        if self.process_num == 1:
            run_result = [
                self._run_interaction(
                    interaction,
                    self._get_interaction_specific_data(interaction),
                    self.outcome_variable,
                    self.covariates,
                    self.min_n,
                    self.family,
                    self.use_t,
                    self.report_betas,
                )
                for interaction in self.interactions
            ]

        else:
            with multiprocessing.Pool(processes=self.process_num) as pool:
                run_result = pool.starmap(
                    self._run_interaction,
                    zip(
                        self.interactions,
                        [
                            self._get_interaction_specific_data(interaction)
                            for interaction in self.interactions
                        ],
                        repeat(self.outcome_variable),
                        repeat(self.covariates),
                        repeat(self.min_n),
                        repeat(self.family),
                        repeat(self.use_t),
                        repeat(self.report_betas),
                    ),
                )

        for interaction, interaction_result in zip(self.interactions, run_result):
            interaction_str = ":".join(interaction)
            results, warnings, error = interaction_result
            self.results.extend(results)  # Merge lists into one list
            self.warnings[interaction_str] = warnings
            if error is not None:
                self.errors[interaction_str] = error

        click.echo(
            click.style(
                f"\tFinished Running {len(self.interactions):,} interactions",
                fg="green",
            )
        )

        self.run_complete = True

    @classmethod
    def _run_interaction(
        cls,
        interaction: Tuple[str, str],
        data: pd.DataFrame,
        outcome_variable: str,
        covariates: List[str],
        min_n: int,
        family: str,
        use_t: bool,
        report_betas: bool,
    ):
        i1, i2 = interaction

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

            # GITHUB/ISSUES 116: Regression matrix with empty categories
            # (Moved after to clear NAN)
            # Remove unused categories (warning when this occurs)
            # removed_cats = _remove_empty_categories(data)
            # if len(removed_cats) >= 1:
            #     for extra_cat_var, extra_cats in removed_cats.items():
            #         warnings_list.append(
            #             f"'{str(extra_cat_var)}'"
            #             f" had categories with no occurrences: "
            #             f"{', '.join([str(c) for c in extra_cats])} "
            #             f"after removing observations with missing values"
            #         )

            # Get the formulas
            # Restricted Formula - covariates and main effects
            formula_restricted = f"Q('{outcome_variable}') ~ 1 + Q('{i1}') + Q('{i2}')"
            if len(varying_covars) > 0:
                formula_restricted += " + "
                formula_restricted += " + ".join([f"Q('{v}')" for v in varying_covars])
            # Full Formula - restricted plus interactions
            formula = formula_restricted + f" + Q('{i1}'):Q('{i2}')"

            # Apply the complete_case_mask to the data to ensure categorical models use the same data in the LRT
            data = data.loc[complete_case_mask]

            # GITHUB/ISSUES 116: Regression matrix with empty categories
            # Remove unused categories (warning when this occurs)
            removed_cats = _remove_empty_categories(data)
            if len(removed_cats) >= 1:
                for extra_cat_var, extra_cats in removed_cats.items():
                    warnings_list.append(
                        f"'{str(extra_cat_var)}'"
                        f" had categories with no occurrences: "
                        f"{', '.join([str(c) for c in extra_cats])} "
                        f"after removing observations with missing values"
                    )

            # Run Regression LRT Test
            for regression_result in cls._run_interaction_regression(
                data,
                formula,
                formula_restricted,
                family,
                use_t,
                report_betas,
                i1,
                i2,
            ):
                result = cls._get_default_result_dict(i1, i2, outcome_variable)
                result["N"] = N
                # TODO:
                result.update(regression_result)
                result_list.append(result)

        except Exception as e:
            error = str(e)
            if result is None:
                result_list = [cls._get_default_result_dict(i1, i2, outcome_variable)]

        return result_list, warnings_list, error
