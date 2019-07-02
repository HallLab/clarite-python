import pandas as pd

from .. import io


@pd.api.extensions.register_dataframe_accessor("clarite_io")
class ClariteIODFAccessor(object):
    """
    Available as 'clarite_io'
    """
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        # TODO: Perform any required validation
        if False:
            raise AttributeError("")

    def save_dtypes(self, filename: str):
        """
        Save a datatype file (.dtype) for the given data

        Parameters
        ----------
        filename: str
            Name of data file to be used in CLARITE - the 'dtypes' extension is added automatically if needed.

        Returns
        -------
        None

        Examples
        --------
        >>> df.clarite_io.save_dtypes('data/test_data')
        """
        df = self._obj
        io.save_dtypes(data=df, filename=filename)

    def save(self, filename: str):
        """
        Save a data to a file along with a dtype file

        Parameters
        ----------
        data: pd.DataFrame
            Data to be saved
        filename: str
            File with data to be used in CLARITE

        Returns
        -------
        None

        Examples
        --------
        >>> df.clarite_io.save('data/test_data')
        """
        df = self._obj
        io.save(data=df, filename=filename)
