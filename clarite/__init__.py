try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

from .modules import modify, plot, describe, analyze, load, survey

__version__ = importlib_metadata.version(__name__)

__all__ = ["__version__", "modify", "plot", "describe", "analyze", "load", "survey"]
