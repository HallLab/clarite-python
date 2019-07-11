=====
Usage
=====

Organization of Functions
-------------------------

CLARITE has many functions organized into several different modules:

Analyze
  Functions related to calculating EWAS results

Describe
  Functions used to gather information about data

IO
  Functions used to load, convert, and save data

Modify
  Functions used to filter and/or modify data

Plot 
  Functions that generate plots

Process 
  Functions used to combine datasets together, transfer data between datasets, or split datasets into multiple parts

Survey
  Functions and classes related to handling data with a complex survey design


Coding Style
------------
There are three primary ways of using CLARITE'.

1. Using the CLARITE package as part of a python script or Jupyter notebook

This can be done using the function directly:

.. code-block:: python

   import clarite
   df = clarite.io.load_data('data.txt')
   df_filtered = clarite.modify.colfilter_min_n(df, n=250)
   df_filtered_complete = clarite.modify.rowfilter_incomplete_obs(df_filtered)
   clarite.plot.distributions(df_filtered_complete, filename='plots.pdf')

Or it can be done using Pandas *pipe*

.. code-block:: python

   clarite.plot.distributions(df.pipe(clarite.modify.colfilter_min_n, n=250)\
                                .pipe(clarite.modify.rowfilter_incomplete_obs),
                              filename='plots.pdf')

2. Using the command line tool

.. code-block:: bash

   clarite-cli modify colfilter-min-n data.txt data_filtered.txt -n 250
   clarite-cli modify rowfilter-incomplete-obs data_filtered.txt data_filtered_complete.txt
   clarite-cli plot distributions data_filtered_complete.txt plots.pdf

3. Using the GUI (coming soon) 
