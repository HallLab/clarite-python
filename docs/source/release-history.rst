===============
Release History
===============

v1.2.0 (2021-03-12)
-------------------

Enhancements
^^^^^^^^^^^^
* Add clariate.analyze.interaction_test
* Improved logging in r_survey ewas
* Refactored lots of regression and ewas code to make it more efficient and provide more validation of input data, including better handling of variable names with symbols/numbers
* Corrected instructions on installing R packages with Conda
* Improved documentation of Regression classes
* Manhattan plots have a "return_figure" option

Fixes
^^^^^
* r_survey regression no longer uses an LRT for binary variables in order to make it concordant with regression in python
* outlier_removal is now working as intended

Tests
^^^^^
* Added a test for outlier removal


v1.1.1 (2020-09-12)
-------------------

Fixes
^^^^^
* Fixed a failing test caused by newer dependency versions

v1.1.0 (2020-08-14)
-------------------

Enhancements
^^^^^^^^^^^^
* Add a `subset` method on the SurveyDesignSpec class
* Refactored regression so that the `ewas` function now takes a `regression_kind` parameter

Tests
^^^^^
* Added tests for the `subset` method

v1.0.1 (2020-06-12)
-------------------

Enhancements
^^^^^^^^^^^^
* Improve the legend in the top_results plot and add additional parameters similar to the manhattan plots

Fixes
^^^^^
* Update the default names for the ewas parameter *single_cluster* in the CLI
* Add the "drop_unweighted" parameter to the printed result of Survey Designs
* Fix an IndexError caused by non-continuous variables being passed to describe.skewness
* Fix the travis build (the bioconda channel must be specified to install r-survey)

Tests
^^^^^
* Added a plot test for passing "None" as the cutoff to the top results plot

v1.0.0 (2020-06-04)
-------------------

Fixes
^^^^^
* Fixed *ewas_r* not working for some parameter combinations
* Improved the *top_results* plot to work with non-continuous values (which don't have Betas)
* Corrected ewas results for some scenarios (strata and clusters) related to missing data (incorrect degrees of freedom)

Tests
^^^^^
* Added additional analysis tests with realistic data (more missing values)
* All analysis tests are now passing with 1E-4 relative tolerance
* Added the first plot tests


v0.10.0 (2020-05-28)
--------------------

Enhancements
^^^^^^^^^^^^
* Manhattan plot split into three functions (raw, bonferroni, and fdr) and now has a custom threshold parameter
* Use Pandas v1.0+
* Refactored regression objects to simplify internal code and potentially allow for more types of regression in the future
* Added an ewas_r function that seamlessly runs the ewas analysis in R, using the R *survey* library
  * This is recommended when using weights, as the python version has some inconsistencies in some edge cases
* Added a skewness function
* Added a *top_results* plot
* Add a *drop_unweighted* parameter to the *SurveyDesignSpec* to provide an easy (if potentially incorrect) workaround for observations with missing weights

Fixes
^^^^^
* Provide a warning and a convenience function when categorical types have categories with no occurrences
* Catch errors when categorizing variables with many unique string values
* Corrected some edge-case EWAS results when using weights in the presence of missing values
* Avoid some cryptic errors by ensuring the input to some functions is a DataFrame and not a Series

Tests
^^^^^
Many additional tests were added, especially related to EWAS


v0.9.1 (2019-11-20)
-------------------

Minor documentation update

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
