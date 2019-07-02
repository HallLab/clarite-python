import pandas as pd

from .. import describe


@pd.api.extensions.register_dataframe_accessor("clarite_describe")
class ClariteDescribeDFAccessor(object):
    """
    Available as 'clarite_describe'
    """
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        # TODO: Perform any required validation
        if False:
            raise AttributeError("")

    def correlations(self, threshold: float = 0.75):
        """
        Return variables with pearson correlation above the threshold

        Parameters
        ----------
        threshold: float, between 0 and 1
            Return a dataframe listing pairs of variables whose absolute value of correlation is above this threshold

        Returns
        -------
        result: pd.DataFrame
            DataFrame listing pairs of correlated variables and their correlation value

        Examples
        --------
        >>> correlations = df.clarite_describe.correlations(threshold=0.9)
        >>> correlations.head()
                           var1      var2  correlation
        36704  supplement_count  DSDCOUNT     1.000000
        32807          DR1TM181  DR1TMFAT     0.997900
        33509          DR1TP182  DR1TPFAT     0.996172
        39575          DRD370FQ  DRD370UQ     0.987974
        35290          DR1TS160  DR1TSFAT     0.984733
        """
        df = self._obj
        return describe.correlations(df, threshold=threshold)

    def freq_table(self):
        """
        Return the count of each unique value for all categorical variables.  Non-categorical typed variables
        will return a single row with a value of '<Non-Categorical Values>' and the number of non-NA values.

        Returns
        -------
        result: pd.DataFrame
            DataFrame listing variable, value, and count for each categorical variable

        Examples
        --------
        >>> df.clarite_describe.freq_table().head(n=10)
           variable value  count
        0                 SDDSRVYR                         2   4872
        1                 SDDSRVYR                         1   4191
        2                   female                         1   4724
        3                   female                         0   4339
        4  how_many_years_in_house                         5   2961
        5  how_many_years_in_house                         3   1713
        6  how_many_years_in_house                         2   1502
        7  how_many_years_in_house                         1   1451
        8  how_many_years_in_house                         4   1419
        9                  LBXPFDO  <Non-Categorical Values>   1032
        """
        df = self._obj
        return describe.freq_table(df)

    def percent_na(self):
        """
        Return the percent of observations that are NA for each variable

        Returns
        -------
        result: pd.Series
            Series listing percent NA for each variable

        Examples
        --------
        >>> df.clarite_describe.percent_na()
        SDDSRVYR                 0.000000
        female                   0.000000
        LBXHBC                   0.049321
        LBXHBS                   0.049873
        """
        df = self._obj
        return describe.percent_na(df)
