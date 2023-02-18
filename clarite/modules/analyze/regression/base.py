from abc import ABCMeta, abstractmethod
from collections import defaultdict
from typing import List, Optional, Tuple

import click
import pandas as pd

from clarite.internal.utilities import _get_dtypes, _remove_empty_categories


class Regression(metaclass=ABCMeta):
    """
    Abstract Base Class for Regression objects used in EWAS.

    Parameters
    ----------
    data: pd.DataFrame
        Data used in the analysis
    outcome_variable: str
        The variable to be used as the output (y) of the regression(s)
    regression_variables: List[str]
        Variables to be regressed
    covariates: List[str], optional
        The variables to be used as covariates in each regression.
        Any variables in the DataFrames not listed as covariates are regressed.
        Use `None` or an empty list when no covariates are being used.

    Notes
    -----
    These are the abstract methods:
    * run() -> None
    * get_results() -> pd.DataFrame
    """

    def __init__(
        self,
        data: pd.DataFrame,
        outcome_variable: str,
        regression_variables: List[str],
        covariates: Optional[List[str]] = None,
    ):
        # Copy the data to avoid changing the original.  The copy will be modified in-place.
        data = data.copy()
        # Print a warning if there are any empty categories and remove them
        # This is done to distinguish from those that become missing during analysis (and could be an issue)
        empty_categories = _remove_empty_categories(data)
        if len(empty_categories) > 0:
            warning = (
                f"Warning: {len(empty_categories)} variables had empty categories (the category exists in the"
                f" data type, but doesn't occur in any observations):"
            )
            for extra_cat_var, extra_cats in empty_categories.items():
                missing_cat = ", ".join([f"'{c}'" for c in extra_cats])
                warning += f"\n\t{extra_cat_var} missing {missing_cat}"
            click.echo(click.style(warning, fg="yellow"))
        # Store minimal regression parameters
        self.data = data
        self.outcome_variable = outcome_variable
        if covariates is None:
            covariates = []
        self.covariates = covariates
        # Validate parameters
        self.outcome_dtype = None
        self.regression_variables = dict()  # Mapping dtype to the variable names
        self._validate_regression_params(regression_variables)
        # Store defaults/placeholders (mapping each regressed variable to the value
        self.results = (
            list()
        )  # List of dictionaries that get transformed into DataFrame rows
        self.errors = dict()
        self.warnings = defaultdict(list)
        # Flag marking whether self.run() was called, determining if it is valid to use self.get_results()
        self.run_complete = False

        # Description is the string representation of self that is updated as the object is initialized
        self.description = ""

    def __str__(self):
        return (
            f"{self.__class__.__name__}\n"
            + ("-" * 25)
            + f"\n{self.description}\n"
            + ("-" * 25)
        )

    def _validate_regression_params(self, regression_variables):
        """
        Validate standard regression parameters- data, outcome_variable, and covariates.  Store relevant information.
        """
        # Covariates must be a list
        if type(self.covariates) != list:
            raise ValueError("'covariates' must be specified as a list or set to None")

        # Make sure the index of each dataset is not a multiindex and give it a consistent name
        if isinstance(self.data.index, pd.MultiIndex):
            raise ValueError("Data must not have a multiindex")
        if self.data.index.name != "ID":
            click.echo(
                click.style(
                    "The index name in the provided data is not 'ID'. Was it loaded using clarite.load?",
                    fg="yellow",
                )
            )
            self.data.index.name = "ID"

        # Collect lists of regression variables
        types = _get_dtypes(self.data)
        rv_types = {v: t for v, t in types.items() if v in regression_variables}
        rv_count = 0
        for dtype in ["binary", "categorical", "continuous", "genotypes"]:
            self.regression_variables[dtype] = [
                v for v, t in rv_types.items() if t == dtype
            ]
            rv_count += len(self.regression_variables[dtype])

        # Ensure there are variables which can be regressed
        if rv_count == 0:
            raise ValueError("No variables are available to run regression on")

        # Ensure covariates are all present and not unknown type
        self.covariate_types = {
            covariate: types.get(covariate, None) for covariate in self.covariates
        }
        missing_covariates = [c for c, dt in self.covariate_types.items() if dt is None]
        unknown_covariates = [
            c for c, dt in self.covariate_types.items() if dt == "unknown"
        ]
        if len(missing_covariates) > 0:
            raise ValueError(
                f"One or more covariates were not found in the data: {', '.join(missing_covariates)}"
            )
        if len(unknown_covariates) > 0:
            raise ValueError(
                f"One or more covariates have an unknown datatype: {', '.join(unknown_covariates)}"
            )

        # Raise an error if the outcome variable dtype isn't compatible with regression in general
        self.outcome_dtype = types.get(self.outcome_variable, None)
        if self.outcome_variable in self.covariates:
            raise ValueError(
                f"The outcome variable ('{self.outcome_variable}') cannot also be a covariate."
            )
        elif self.outcome_dtype is None:
            raise ValueError(
                f"The outcome variable ('{self.outcome_variable}') was not found in the data."
            )
        elif self.outcome_dtype == "unknown":
            raise ValueError(
                f"The outcome variable ('{self.outcome_variable}') has an unknown type."
            )
        elif self.outcome_dtype == "constant":
            raise ValueError(
                f"The outcome variable ('{self.outcome_variable}') is a constant value."
            )

    def _log_errors_and_warnings(self):
        """Print any errors and warnings present in the regression (if any)"""
        # Log errors
        if len(self.errors) == 0:
            click.echo(click.style("0 tests had an error", fg="green"))
        elif len(self.errors) > 0:
            click.echo(
                click.style(f"{len(self.errors):,} tests had an error", fg="red")
            )
            for label, error in self.errors.items():
                click.echo(click.style(f"\t{label} = NULL due to: {error}", fg="red"))
        # Log warnings
        for label, warning_list in self.warnings.items():
            if len(warning_list) > 0:
                click.echo(click.style(f"{label} had warnings:", fg="yellow"))
                for warning in warning_list:
                    click.echo(click.style(f"\t{warning}", fg="yellow"))

    @staticmethod
    def _check_covariate_values(
        data, covariates, keep_row_mask
    ) -> Tuple[List[str], List[str]]:
        """Remove covariates that do not vary, warning when this occurs"""
        warnings = []
        unique_values = data.loc[keep_row_mask, covariates].nunique()
        varying_covars = list(unique_values[unique_values > 1].index.values)
        non_varying_covars = list(unique_values[unique_values <= 1].index.values)
        if len(non_varying_covars) > 0:
            warnings.append(
                f"non-varying covariates(s): {', '.join(non_varying_covars)}"
            )
        return varying_covars, warnings

    @abstractmethod
    def run(self) -> None:
        """Run the regression"""

    @abstractmethod
    def get_results(self) -> pd.DataFrame:
        """Return results of the regression as a DataFrame"""
