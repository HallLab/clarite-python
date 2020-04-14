============
Installation
============

Basic Install
-------------
At the command line::

    $ pip install clarite


Running R code from CLARITE
---------------------------

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

