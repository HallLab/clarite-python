=====
Usage
=====

Start by importing CLARITE.

.. code-block:: python

    import clarite

Structure
#########

Clarite consists of several functions and classes, including a Pandas DataFrame Accessor.  The DataFrame Accessor adds methods to DataFrame objects under a `clarite` namespace:

.. code-block:: python

    df = clarite.io.load_data('nhanes_binary.txt', index_col="ID", sep="\t")
    df = df.clarite.make_bin()


DataFrame Accessor functions
############################

Column (Variable) Filters
*************************

.. automethod:: clarite.ClariteDataframeAccessor.colfilter_percent_zero

.. automethod:: clarite.ClariteDataframeAccessor.colfilter_min_n

.. automethod:: clarite.ClariteDataframeAccessor.colfilter_min_cat_n

Row (Observation) Filters
*************************

.. automethod:: clarite.ClariteDataframeAccessor.rowfilter_incomplete_observations

Plotting
******************

.. automethod:: clarite.ClariteDataframeAccessor.plot_hist

.. automethod:: clarite.ClariteDataframeAccessor.plot_manhattan

Other DataFrame Accessor Functions
**********************************

.. automethod:: clarite.ClariteDataframeAccessor.categorize

EWAS Functions
##############

.. autofunction:: clarite.ewas

.. autofunction:: clarite.add_corrected_pvalues

Utilities
#########

.. autofunction:: clarite.make_bin

.. autofunction:: clarite.make_categorical

.. autofunction:: clarite.make_continuous

IO Functions
############

.. autofunction:: clarite.io.load_data

Classes
#######

.. autoclass:: clarite.SurveyDesignSpec
