"""
Analyze
========

Functions used for analyses such as EWAS

  .. autofunction:: association_study
  .. autofunction:: interaction_study
  .. autofunction:: add_corrected_pvalues

"""

from .association_study import association_study
from .ewas import ewas
from .interaction_study import interaction_study
from .utils import add_corrected_pvalues
from . import regression

__all__ = [
    "association_study",
    "ewas",
    "interaction_study",
    "add_corrected_pvalues",
    "regression",
]

# Constants
required_result_columns = {"N", "pvalue", "error", "warnings"}
result_columns = [
    "Variable_type",
    "Converged",
    "N",
    "Beta",
    "SE",
    "Beta_pvalue",
    "LRT_pvalue",
    "Diff_AIC",
    "pvalue",
]
corrected_pvalue_columns = ["pvalue_bonferroni", "pvalue_fdr"]

__all__.append("required_result_columns")
__all__.append("result_columns")
__all__.append("corrected_pvalue_columns")
