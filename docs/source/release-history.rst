===============
Release History
===============

v0.5.0 (2019-06-28)
-------------------

Enhancements
^^^^^^^^^^^^
* Added a function to recode values - https://github.com/HallLab/clarite-python/issues/4
* Added a function to filter outlier values - https://github.com/HallLab/clarite-python/issues/5
* Added a function to generate manhattan plots for multiple datasets together - https://github.com/HallLab/clarite-python/issues/9

Fixes
^^^^^
* Add some validation of input DataFrames to prevent some errors in calculations

Tests
^^^^^
* Added an initial batch of tests

v0.4.0 (2019-06-18)
-------------------
Support EWAS with binary outcomes.
Additional handling of NA values in covariates and the phenotype.
Add a 'min_n' parameter to the ewas function to require a minimum number of observations after removing incomplete cases.
Add additional functions including 'plot_distributions', 'merge_variables', 'get_correlations', 'get_freq_table', and 'get_percent_na'

v0.3.0 (2019-05-31)
-------------------
Add support for complex survey designs

v0.2.1 (2019-05-02)
-------------------
Added documentation for existing functions

v0.2.0 (2019-04-30)
-------------------
First functional version.  Mutliple methods are available under a 'clarite' Pandas accessor.

v0.1.0 (2019-04-23)
-----------------------------------
Initial Release
