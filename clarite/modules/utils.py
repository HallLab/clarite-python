"""
Utils
======

Miscellaneous utility functions

  .. autosummary::
     :toctree: modules/utils

     setup_r_packages
"""

from pathlib import Path


def setup_r_packages():
    """
    Installs r packages used by CLARITE if they are not already installed.

    Packages
    --------
    * survey
    """
    import rpy2.robjects as ro
    r_code_folder = (Path(__file__).parent.parent / 'r_code')
    filename = str(r_code_folder / "setup_r_packages.R")
    ro.r.source(filename)
