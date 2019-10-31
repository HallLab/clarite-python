===============
Release History
===============

v0.9.0 (2019-10-31)
-------------------

Enhancements
^^^^^^^^^^^^
* Add a *figure* parameter to histogram and manhattan plots in order to plot to an existing figure
* *SurveyDesignSpec* can now utilize more parameters, such as *fpc*
* The larger (numeric or alphabetic) binary variable is always treated as the success case for binary phenotypes
* Improved logging during EWAS, including printing the survey design information
* Extensively updated documentation
* CLARITE now has a logo!

Fixes
^^^^^
* Corrected an indexing error that sometimes occurred when removing rows with missing weights
* Improve precision in EWAS results for weighted analyses by using sf instead of 1-cdf
* Change some column names in the EWAS output to be more clear

Tests
^^^^^
An R script and the output of that script is now included.  The R output is compared to the python output in the
test suite in order to ensure analysis result concordance between R and Python for several analysis scenarios.

v0.8.0 (2019-09-03)
-------------------

Enhancements
^^^^^^^^^^^^
* Allow file input in the command line for skip/only
* Make the manhattan plot function less restrictive of the data passed into it
* Use skip/only in the transform function

Fixes
^^^^^
* Categorization would silently fail if there was only one variable of a given type


v0.7.0 (2019-07-23)
-------------------

Enhancements
^^^^^^^^^^^^
* Improvements to the CLI and printed log messages.
* The functions from the 'Process' module were put into the 'Modify' module.
* Datasets are no longer split apart when categorizing.

v0.6.0 (2019-07-11)
-------------------

Extensive changes in organization, but limited new functionality (not counting the CLI).

Enhancements
^^^^^^^^^^^^
* Reorganize functions - https://github.com/HallLab/clarite-python/pull/13
* Add a CLI - https://github.com/HallLab/clarite-python/pull/11

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
