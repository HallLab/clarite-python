# flake8: noqa
"""

What is a DataFrame Accessor?
=============================

Pandas DataFrame Accessors attach methods to the DataFrame namespace, providing a different way to access many CLARITE functions

.. code-block:: python
    
    clarite.modify.colfilter_min_n(df, n=250)

can be written as:

.. code-block:: python
    
    df.clarite_modify.colfilter_min_n(n=250)

Available DataFrame Accessors
=============================

  .. autosummary::
     :toctree: modules/df_accessors

     ClariteDescribeDFAccessor
     ClariteModifyDFAccessor
     ClaritePlotDFAccessor
     ClariteProcessDFAccessor
"""

from .clarite_describe import ClariteDescribeDFAccessor
from .clarite_modify import ClariteModifyDFAccessor
from .clarite_plot import ClaritePlotDFAccessor
from .clarite_process import ClariteProcessDFAccessor