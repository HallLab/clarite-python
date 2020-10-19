from abc import ABCMeta, abstractmethod
from collections import defaultdict
from typing import List, Optional

import click
import pandas as pd

from clarite.internal.utilities import _get_dtypes, _remove_empty_categories


class Regression(metaclass=ABCMeta):
    """
    Abstract Base Class for Regression objects used in EWAS.

    Minimum Parameters
    ------------------
    data: pd.DataFrame
        Data used in the analysis
    outcome_variable: str
        The variable to be used as the output (y) of the regression(s)
    covariates: List[str], optional
        The variables to be used as covariates in each regression.
        Any variables in the DataFrames not listed as covariates are regressed.
        Use `None` or an empty list when no covariates are being used.

    Abstract Methods
    ----------------
    run() -> None
    get_results() -> pd.DataFrame
    """
    def __init__(self,
                 data: pd.DataFrame,
                 outcome_variable: str,
                 covariates: Optional[List[str]] = None):
        # Print a warning if there are any empty categories and remove them
        # This is done to distinguish from those that become missing during analysis (and could be an issue)
        empty_categories = _remove_empty_categories(data)
        if len(empty_categories) > 0:
            warning = f"Warning: {len(empty_categories)} variables had empty categories (the category exists in the" \
                      f" data type, but doesn't occur in any observations):"
            for extra_cat_var, extra_cats in empty_categories.items():
                missing_cat = ", ".join([f"'{c}'" for c in extra_cats])
                warning += f"\n\t{extra_cat_var} missing {missing_cat}"
            click.echo(click.style(warning, fg='yellow'))
        # Store minimal regression parameters
        self.data = data
        self.outcome_variable = outcome_variable
        if covariates is None:
            covariates = []
        self.covariates = covariates
        # Validate parameters
        self.outcome_dtype = None
        self.regression_variables = dict()  # Mapping dtype to the variable names
        self.validate_regression_params()
        # Store defaults/placeholders (mapping each regressed variable to the value
        self.results = defaultdict(dict)
        self.errors = dict()
        self.warnings = defaultdict(list)
        # Flag marking whether self.run() was called, determining if it is valid to use self.get_results()
        self.run_complete = False

        # Description is the string representation of self that is updated as the object is initialized
        self.description = ""

    def __str__(self):
        return f"{self.__class__.__name__}\n" + ("-"*25) + f"\n{self.description}\n" + ("-"*25)

    def validate_regression_params(self):
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
            click.echo(click.style("The index name in the provided data is not 'ID'. Was it loaded using clarite.load?",
                                   fg='yellow'))
            self.data.index.name = "ID"

        # Collect lists of regression variables
        types = _get_dtypes(self.data)
        rv_types = {v: t for v, t in types.iteritems() if v not in self.covariates and v != self.outcome_variable}
        rv_count = 0
        for dtype in ['binary', 'categorical', 'continuous']:
            self.regression_variables[dtype] = [v for v, t in rv_types.items() if t == dtype]
            rv_count += len(self.regression_variables[dtype])

        # Ensure there are variables which can be regressed
        if rv_count == 0:
            raise ValueError("No variables are available to run regression on")

        # Ensure covariates are all present and not unknown type
        covariate_types = [types.get(c, None) for c in self.covariates]
        missing_covariates = [c for c, dt in zip(self.covariates, covariate_types) if dt is None]
        unknown_covariates = [c for c, dt in zip(self.covariates, covariate_types) if dt == 'unknown']
        if len(missing_covariates) > 0:
            raise ValueError(f"One or more covariates were not found in the data: {', '.join(missing_covariates)}")
        if len(unknown_covariates) > 0:
            raise ValueError(f"One or more covariates have an unknown datatype: {', '.join(unknown_covariates)}")

        # Raise an error if the outcome variable dtype isn't compatible with regression in general
        self.outcome_dtype = types.get(self.outcome_variable, None)
        if self.outcome_variable in self.covariates:
            raise ValueError(f"The outcome variable ('{self.outcome_variable}') cannot also be a covariate.")
        elif self.outcome_dtype is None:
            raise ValueError(f"The outcome variable ('{self.outcome_variable}') was not found in the data.")
        elif self.outcome_dtype == 'unknown':
            raise ValueError(f"The outcome variable ('{self.outcome_variable}') has an unknown type.")
        elif self.outcome_dtype == 'constant':
            raise ValueError(f"The outcome variable ('{self.outcome_variable}') is a constant value.")

    @abstractmethod
    def run(self) -> None:
        """Run the regression"""

    @abstractmethod
    def get_results(self) -> pd.DataFrame:
        """Return results of the regression as a DataFrame"""
