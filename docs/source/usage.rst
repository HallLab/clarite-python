=====
Usage
=====

Start by importing CLARITE.

.. code-block:: python

    import clarite

Organization of Functions
-------------------------

Most of the functions in CLARITE take a DataFrame as the first parameter which is assumed to have a different variable in each column and a different observation in each row (using a consistent index).  These are largely organized into 4 modules:

Describe
  Functions used to gather information about some data

Modify
  Functions used to filter and/or change some data

Process 
  Functions used to process data from one form into another, such as categorizing variables and placing them in separate DataFrames 

Plot 
  Functions that generate plots

Coding Style
------------
Inspired by `PyJanitor <https://pyjanitor.readthedocs.io>`_, there are three primary ways of using these functions in the CLARITE package:

1. Using the DataFrame Accessors

.. code-block:: python

   df.clarite_modify.colfilter_min_n(n=250)\
     .clarite_modify.rowfilter_incomplete_observations()\
     .clarite_plot.distributions(filename='plots.pdf')

2. Using the functional API

.. code-block:: python

   df_filtered = clarite.modify.colfilter_min_n(df, n=250)
   df_filtered_complete = clarite.modify.rowfilter_incomplete_observations(df_filtered)
   clarite.plot.distributions(df_filtered_complete, filename='plots.pdf')

3. Using a Pandas *pipe*

.. code-block:: python

   clarite.plot.distributions(df.pipe(clarite.modify.colfilter_min_n, n=250)\
                                .pipe(clarite.modify.rowfilter_incomplete_observations),
                              filename='plots.pdf')


Other modules
-------------

Some functions do not take a single DataFrame as the first parameter- these are not available in DataFrame Accessors, but are available as functions in these modules:

Survey
    Complex survey design
IO
    Input/Output of data in different formats
Analyze
    EWAS and associated calculations
