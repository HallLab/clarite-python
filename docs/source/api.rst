=============
API Reference
=============

Modules
-------

.. automodule:: clarite.analyze
  :members: ewas, add_corrected_pvalues

-----

.. automodule:: clarite.describe
  :members: correlations, freq_table, percent_na

-----

.. automodule:: clarite.io
  :members: load_data

-----

.. automodule:: clarite.modify
  :members: colfilter_percent_zero, colfilter_min_n, colfilter_min_cat_n, rowfilter_incomplete_observations,
            recode_values, remove_outliers, make_binary, make_categorical, make_continuous, merge_variables

-----

.. automodule:: clarite.plot
  :members:

-----

.. automodule:: clarite.process
   :members:

-----

.. automodule:: clarite.survey
   :members: SurveyDesignSpec

DataFrame Accessors
-------------------

.. autoclass:: clarite.ClariteDescribeDFAccessor
  :members:

----

.. autoclass:: clarite.ClariteModifyDFAccessor
  :members:
  
----

.. autoclass:: clarite.ClaritePlotDFAccessor
  :members:
  
----

.. autoclass:: clarite.ClariteProcessDFAccessor
  :members: