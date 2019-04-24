# flake8: noqa
from ._version import get_versions

from .colfilters import ColFilterAccessor
from .rowfilters import RowFilterAccessor

__version__ = get_versions()['version']
del get_versions
