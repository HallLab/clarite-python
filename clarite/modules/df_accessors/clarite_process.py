import pandas as pd

from .. import process


@pd.api.extensions.register_dataframe_accessor("clarite_process")
class ClariteProcessDFAccessor(object):
    """Available as 'clarite_process'"""
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        # TODO: Perform any required validation
        if False:
            raise AttributeError("")

    def categorize(self, cat_min: int = 3, cat_max: int = 6, cont_min: int = 15):
        """
        Divide variables into binary, categorical, continuous, and ambiguous dataframes

        Parameters
        ----------
        cat_min: int, default 3
            Minimum number of unique, non-NA values for a categorical variable
        cat_max: int, default 6
            Maximum number of unique, non-NA values for a categorical variable
        cont_min: int, default 15
            Minimum number of unique, non-NA values for a continuous variable

        Returns
        -------
        bin_df: pd.DataFrame
            DataFrame with variables that were categorized as *binary*
        cat_df: pd.DataFrame
            DataFrame with variables that were categorized as *categorical*
        bin_df: pd.DataFrame
            DataFrame with variables that were categorized as *continuous*
        other_df: pd.DataFrame
            DataFrame with variables that were not categorized and should be examined manually

        Examples
        --------
        >>> nhanes_bin, nhanes_cat, nhanes_cont, nhanes_other = nhanes.clarite_process.categorize()
        10 of 945 variables (1.06%) had no non-NA values and are discarded.
        33 of 945 variables (3.49%) had only one value and are discarded.
        361 of 945 variables (38.20%) are classified as binary (2 values).
        44 of 945 variables (4.66%) are classified as categorical (3 to 6 values).
        461 of 945 variables (48.78%) are classified as continuous (>= 15 values).
        36 of 945 variables (3.81%) are not classified (between 6 and 15 values).
        """
        df = self._obj
        return process.categorize(
            df, cat_min=cat_min, cat_max=cat_max, cont_min=cont_min
        )
