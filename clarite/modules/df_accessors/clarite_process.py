from typing import List, Optional, Union

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

    def merge_variables(self, other: pd.DataFrame, how: str = 'outer'):
        """
        Merge a list of dataframes with different variables side-by-side.  Keep all observations ('outer' merge) by default.

        Parameters
        ----------
        other: pd.DataFrame
            "right" DataFrame which uses the same index
        how: merge method, one of {'left', 'right', 'inner', 'outer'}
            Keep only rows present in the original data, the merging data, both datasets, or either dataset.

        Examples
        --------
        >>> import clarite
        >>> df = df_bin.clarite_process.merge_variables(df_cat)
        """
        df = self._obj
        return process.merge_variables(df, other, how=how)

    def move_variables(self, other: pd.DataFrame,
                       skip: Optional[Union[str, List[str]]] = None, only: Optional[Union[str, List[str]]] = None):
        """
        Move one or more variables from this DataFrame to another

        Parameters
        ----------
        other: pd.DataFrame
            DataFrame (which uses the same index) that the variable(s) will be moved to
        skip: str, list or None (default is None)
            List of variables that will *not* be moved
        only: str, list or None (default is None)
            List of variables that are the *only* ones to be moved

        Returns
        -------
        data: pd.DataFrame
            The first DataFrame with the variables removed
        other: pd.DataFrame
            The second DataFrame with the variables added

        Examples
        --------
        >>> import clarite
        >>> df_cat, df_cont = df_cat.clarite_process.move_variables(df_cont, only=["DRD350AQ", "DRD350DQ", "DRD350GQ"])
        Moved 3 variables.
        >>> discovery_check, discovery_cont = discovery_check.clarite_process.move_variables(discovery_cont)
        Moved 39 variables.
        """
        data = self._obj
        return process.move_variables(data, other=other, skip=skip, only=only)
