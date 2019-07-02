from typing import Optional, List

import pandas as pd

from .. import modify


@pd.api.extensions.register_dataframe_accessor("clarite_modify")
class ClariteModifyDFAccessor(object):
    """Available as 'clarite_modify'"""
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        # TODO: Perform any required validation
        if False:
            raise AttributeError("")

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
        >>> df = df_bin.clarite_modify.merge_variables(df_cat)
        """
        df = self._obj
        return df.merge(other, left_index=True, right_index=True, how=how)

    def colfilter_percent_zero(
        self,
        proportion: float = 0.9,
        skip: Optional[List[str]] = None,
        only: Optional[List[str]] = None,
    ):
        """
        Remove columns which have <proportion> or more values of zero (excluding NA)

        Parameters
        ----------
        proportion: float, default 0.9
            If the proportion of rows in the data with a value of zero is greater than or equal to this value, the variable is filtered out.
        skip: list or None, default None
            List of variables that the filter should *not* be applied to
        only: list or None, default None
            List of variables that the filter should *only* be applied to

        Returns
        -------
        df: pd.DataFrame
            The filtered DataFrame

        Examples
        --------
        >>> nhanes_discovery_cont = nhanes_discovery_cont.clarite_modify.colfilter_percent_zero()
        Removed 30 of 369 variables (8.13%) which were equal to zero in at least 90.00% of non-NA observations.
        """
        df = self._obj
        return modify.colfilter_percent_zero(
            df, proportion=proportion, skip=skip, only=only
        )

    def colfilter_min_n(
        self,
        n: int = 200,
        skip: Optional[List[str]] = None,
        only: Optional[List[str]] = None,
    ):
        """
        Remove columns which have less than <n> unique values (excluding NA)

        Parameters
        ----------
        n: int, default 200
            The minimum number of unique values required in order for a variable not to be filtered
        skip: list or None, default None
            List of variables that the filter should *not* be applied to
        only: list or None, default None
            List of variables that the filter should *only* be applied to

        Returns
        -------
        df: pd.DataFrame
            The filtered DataFrame

        Examples
        --------
        >>> nhanes_discovery_bin = nhanes_discovery_bin.clarite_modify.colfilter_min_n()
        Removed 129 of 361 variables (35.73%) which had less than 200 values
        """
        df = self._obj
        return modify.colfilter_min_n(df, n=n, skip=skip, only=only)

    def colfilter_min_cat_n(
        self,
        n: int = 200,
        skip: Optional[List[str]] = None,
        only: Optional[List[str]] = None,
    ):
        """
        Remove columns which have less than <n> occurences of each unique value

        Parameters
        ----------
        n: int, default 200
            The minimum number of occurences of each unique value required in order for a variable not to be filtered
        skip: list or None, default None
            List of variables that the filter should *not* be applied to
        only: list or None, default None
            List of variables that the filter should *only* be applied to

        Returns
        -------
        df: pd.DataFrame
            The filtered DataFrame

        Examples
        --------
        >>> nhanes_discovery_bin = nhanes_discovery_bin.clarite_modify.colfilter_min_cat_n()
        Removed 159 of 232 variables (68.53%) which had a category with less than 200 values
        """
        df = self._obj
        return modify.colfilter_min_cat_n(df, n=n, skip=skip, only=only)

    def rowfilter_incomplete_observations(
        self, skip: Optional[List[str]] = None, only: Optional[List[str]] = None
    ):
        """
        Remove rows containing null values

        Parameters
        ----------
        skip: list or None, default None
            List of columns that are not checked for null values
        only: list or None, default None
            List of columns that are the only ones to be checked for null values

        Returns
        -------
        df: pd.DataFrame
            The filtered DataFrame

        Examples
        --------
        >>> nhanes = nhanes.clarite_modify.rowfilter_incomplete_observations(only=[phenotype] + covariates)
        Removed 3,687 of 22,624 rows (16.30%) due to NA values in the specified columns
        """
        df = self._obj
        return modify.rowfilter_incomplete_observations(df, skip=skip, only=only)

    def recode_values(
        self,
        replacement_dict,
        skip: Optional[List[str]] = None,
        only: Optional[List[str]] = None,
    ):
        """
        Convert values in a dataframe.  By default, replacement occurs in all columns but this may be modified with 'skip' or 'only'.
        Pandas has more powerful 'replace' methods for more complicated scenarios.

        Parameters
        ----------
        replacement_dict: dictionary
            A dictionary mapping the value being replaced to the value being inserted
        skip: list or None, default None
            List of variables that the replacement should *not* be applied to
        only: list or None, default None
            List of variables that the replacement should *only* be applied to

        Examples
        --------
        >>> nhanes = nhanes.clarite_modify.recode_values({7: np.nan, 9: np.nan}, only=['SMQ077', 'DBD100'])
        Replaced 17 values from 22,624 rows in 2 columns
        >>> nhanes = nhanes.clarite_modify.recode_values({10: 12}, only=['SMQ077', 'DBD100'])
        No occurences of replaceable values were found, so nothing was replaced.
        """
        df = self._obj
        return modify.recode_values(
            df, replacement_dict=replacement_dict, skip=skip, only=only
        )

    def remove_outliers(
        self,
        method: str = "gaussian",
        cutoff=3,
        skip: Optional[List[str]] = None,
        only: Optional[List[str]] = None,
    ):
        """
        Remove outliers from the dataframe by replacing them with np.nan

        Parameters
        ----------
        method: string, 'gaussian' (default) or 'iqr'
            Define outliers using a gaussian approach (standard deviations from the mean) or inter-quartile range
        cutoff: positive numeric, default of 3
            Either the number of standard deviations from the mean (method='gaussian') or the multiple of the IQR (method='iqr')
            Any values equal to or more extreme will be replaced with np.nan
        skip: list or None, default None
            List of variables that the replacement should *not* be applied to
        only: list or None, default None
            List of variables that the replacement should *only* be applied to

        Examples
        --------
        >>> import clarite
        >>> df.clarite_modify.remove_outliers(method='iqr', cutoff=1.5, only=['DR1TVB1', 'URXP07', 'SMQ077'])
        Removing outliers with values < 1st Quartile - (1.5 * IQR) or > 3rd quartile + (1.5 * IQR) in 3 columns
            430 of 22,624 rows of URXP07 were outliers
            730 of 22,624 rows of DR1TVB1 were outliers
            Skipped filtering 'SMQ077' because it is a categorical variable
        >>> df2.clarite_modify.remove_outliers(only=['DR1TVB1', 'URXP07'])
        Removing outliers with values more than 3 standard deviations from the mean in 2 columns
            42 of 22,624 rows of URXP07 were outliers
            301 of 22,624 rows of DR1TVB1 were outliers
        """
        df = self._obj
        return modify.remove_outliers(
            df, method=method, cutoff=cutoff, skip=skip, only=only
        )
