import pandas as pd


@pd.api.extensions.register_dataframe_accessor("rowfilter")
class RowFilterAccessor(object):
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        # TODO: Perform any required validation
        if False:
            raise AttributeError("")

    def remove_incomplete_observations(self, columns):
        """Remove observations(rows) if any of the passed columns is null"""
        invalid_names = set(columns) - set(list(self._obj))
        if len(invalid_names) > 0:
            raise ValueError(f"Invalid column names were provided: {', '.join(invalid_names)}")

        keep_IDs = self._obj[columns].isnull().sum(axis=1) == 0  # Number of NA in each row is 0
        n_removed = len(self._obj) - sum(keep_IDs)

        print(f"Removed {n_removed:,} of {len(self._obj):,} rows ({n_removed/len(self._obj):.2%}) due to NA values in the specified columns")
        return self._obj[keep_IDs]
