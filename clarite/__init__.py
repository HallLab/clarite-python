# flake8: noqa
from ._version import get_versions

from .modules.df_accessors import *
from .modules import process, modify, plot, describe, analyze, io, survey

__version__ = get_versions()['version']
del get_versions
