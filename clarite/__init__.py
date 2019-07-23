# flake8: noqa
from ._version import get_versions

from .modules import modify, plot, describe, analyze, load, survey

__version__ = get_versions()['version']
del get_versions
