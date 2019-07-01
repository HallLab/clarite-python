# flake8: noqa
from ._version import get_versions

from .df_accessor import ClariteProcessDFAccessor, ClariteModifyDFAccessor, ClaritePlotDFAccessor, ClariteDescribeDFAccessor
from .modules import process, modify, plot, describe, analyze
from .other import io, survey

__version__ = get_versions()['version']
del get_versions
