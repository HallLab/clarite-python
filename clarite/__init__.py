# flake8: noqa
from ._version import get_versions

from .clarite import ClariteDataframeAccessor

from.ewas import ewas, add_corrected_pvalues
from .io import load_data

__version__ = get_versions()['version']
del get_versions
