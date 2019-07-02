===============================
CLARITE
===============================

.. image:: https://img.shields.io/travis/HallLab/clarite-python.svg
        :target: https://travis-ci.org/HallLab/clarite-python

.. image:: https://img.shields.io/pypi/v/clarite.svg
        :target: https://pypi.python.org/pypi/clarite


CLeaning to Anlaysis: Reproducibility-based Interface for Traits and Exposures

* Free software: 3-clause BSD license
* Documentation: https://HallLab.github.io/clarite-python.

Usage
--------

Inspired by PyJanitor, there are three ways to run most functions:

1. Using the Pandas DataFrame accessor

.. code-block:: python
    df = (
        df.clarite.modify_colfilter_min_n()
          .clarite.modify_colfilter_percent_zero()
    )   

2. Using the functional API

.. code-block:: python
    from clarite import modify
    df = clarite.modify.colfilter_min_n(df)
    df = clarite.modify.colfilter_percent_zero(df)
  

3. Using the *pipe* method

.. code-block:: python
    from clarite import modify
    df = (
        df.pipe(modify.colfilter_min_n)
          .pipe(modify.colfilter_percent_zero)
        )