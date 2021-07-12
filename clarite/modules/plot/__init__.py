"""
Plot
========

Functions that generate plots

  .. autosummary::
     :toctree: modules/plot

     histogram
     distributions
     manhattan
     manhattan_fdr
     manhattan_bonferroni
     top_results

"""

from .distributions import distributions
from .histogram import histogram
from .manhattan import manhattan, manhattan_fdr, manhattan_bonferroni
from .top_results import top_results

__all__ = [
    "distributions",
    "histogram",
    "manhattan",
    "manhattan_fdr",
    "manhattan_bonferroni",
    "top_results",
]
