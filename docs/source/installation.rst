============
Installation
============

Basic Install
-------------
At the command line::

    $ pip install clarite


Running R code from CLARITE
---------------------------

In order to use the "r_survey" "regression_kind" in the *ewas* function, R must be installed.

  * The version of R must be compatible with the installed version of rpy2 (usually the latest version).
  * The R installation must have the 'survey' package installed: `Rscript -e 'install.packages("survey")'`
  * The `R_HOME` environment variable may need to be set, and `R` must be on the `PATH`.  This can be done in each script:

.. code-block:: python

    import os
    os.environ["R_HOME"] = r"C:\Program Files\R\R-4.0.2"
    os.environ["PATH"]   = r"C:\Program Files\R\R-4.0.2\bin\x64" + ";" + os.environ["PATH"]

Troubleshooting rpy2
^^^^^^^^^^^^^^^^^^^^
From the command line::

  python -m rpy2.situation

Or within a jupyter notebook environment:

.. code-block:: python

    import rpy2.situation
    for row in rpy2.situation.iter_info():
        print(row)
