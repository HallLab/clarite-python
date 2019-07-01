# flake8: noqa
"""
DataFrame Accessors
===================

These provide a different way to access many CLARITE functions from within the DataFrame namespace

.. autosummary::
     :toctree: modules/dataframe_accessors

     ClariteDescribeDFAccessor
     ClariteModifyDFAccessor
     ClaritePlotDFAccessor
     ClariteProcessDFAccessor
"""

from .clarite_describe import ClariteDescribeDFAccessor
from .clarite_modify import ClariteModifyDFAccessor
from .clarite_plot import ClaritePlotDFAccessor
from .clarite_process import ClariteProcessDFAccessor