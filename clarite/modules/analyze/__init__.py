"""
Analyze
========

Functions used for analyses such as EWAS

  .. autosummary::
     :toctree: modules/analyze

     ewas
     interactions
     regression
     add_corrected_pvalues

"""

from .ewas import ewas
from .interactions import interactions
from .utils import add_corrected_pvalues
from . import regression

__all__ = [ewas, interactions, add_corrected_pvalues, regression]

# Constants
required_result_columns = {"N", "pvalue", "error", "warnings"}
result_columns = [
    "Variable_type",
    "Converged",
    "N",
    "Beta",
    "SE",
    "Variable_pvalue",
    "LRT_pvalue",
    "Diff_AIC",
    "pvalue",
]
corrected_pvalue_columns = ["pvalue_bonferroni", "pvalue_fdr"]

__all__.append(required_result_columns)
__all__.append(result_columns)
__all__.append(corrected_pvalue_columns)
