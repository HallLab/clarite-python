# flake8: noqa
from ._version import get_versions

from .modules import process, modify, plot, describe, analyze, io, survey

__version__ = get_versions()['version']
del get_versions
