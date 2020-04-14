===============================
CLARITE
===============================

.. image:: https://img.shields.io/travis/HallLab/clarite-python.svg
        :target: https://travis-ci.org/HallLab/clarite-python

.. image:: https://readthedocs.org/projects/clarite-python/badge/?version=latest
        :target: https://clarite-python.readthedocs.io/en/latest/

.. image:: https://img.shields.io/pypi/v/clarite.svg
        :target: https://pypi.python.org/pypi/clarite

.. image:: docs/source/_static/clarite_logo.png

CLeaning to Analysis: Reproducibility-based Interface for Traits and Exposures
==============================================================================

* Free software: 3-clause BSD license
* Documentation: https://www.hall-lab.org/clarite-python/.

Installation
------------

In order to use the *ewas_r* function, it is recommended to install CLARITE using Conda:

1. Create and activate a conda environment with python 3.6 or 3.7::

    $ conda create -n clarite python=3.7
    $ conda activate clarite
    $ conda config --add channels conda-forge

2. Install rpy2 (optional). CLARITE has a version of the EWAS function that calls R code using the *survey* library::

    $ conda install rpy2

3. Install CLARITE::

    $ pip install clarite

4. Install required R packages (such as *survey*) (optional)::

    $ clarite-cli utils install-r-packages



Citing CLARITE
^^^^^^^^^^^^^^

1.
Lucas AM, et al (2019)
`CLARITE facilitates the quality control and analysis process for EWAS of metabolic-related traits. <https://www.frontiersin.org/article/10.3389/fgene.2019.01240>`_
*Frontiers in Genetics*: 10, 1240

2.
Passero K, et al (2020)
`Phenome-wide association studies on cardiovascular health and fatty acids considering phenotype quality control practices for epidemiological data. <https://www.worldscientific.com/doi/abs/10.1142/9789811215636_0058>`_
*Pacific Symposium on Biocomputing*: 25, 659