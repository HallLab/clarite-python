from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np


class Regression(metaclass=ABCMeta):
    """Abstract Base Class for Regression objects, defining required properties/methods"""
    def __init__(self):
        self.pvalue = np.nan
        self.test_dtype = None
        self.error = None
        self.warnings = list()

    @abstractmethod
    def pre_run_setup(self):
        """Anything that needs to be done after initialization and before running"""
        pass

    @abstractmethod
    def run_continuous(self):
        """Run the regression for a continuous variable"""
        pass

    @abstractmethod
    def run_binary(self):
        """Run the regression for a binary variable"""
        pass

    @abstractmethod
    def run_categorical(self):
        """Run the regression for a categorical variable"""
        pass

    @abstractmethod
    def get_result(self) -> Tuple[Dict, List[str], Optional[str]]:
        """Return a dict of results, a list of warnings, and an optional error"""

    def run(self):
        """Run a regression object, returning the results and logging any warnings/errors"""
        try:
            self.pre_run_setup()
        except Exception as e:
            self.error = str(e)

        # Return early if an error has occurred
        if self.error is not None:
            return self.get_result()

        # Run Regression
        if self.test_dtype == 'continuous':
            self.run_continuous()
        elif self.test_dtype == 'binary':
            self.run_binary()  # Essentially same as continuous, except for the string used to key the results
        elif self.test_dtype == 'categorical':
            self.run_categorical()
        else:
            self.error = f"Unknown regression variable type '{self.test_dtype}'"

        return self.get_result()
