from typing import Dict, List, Optional, Tuple

import pandas as pd

from .modules import modify, process, plot, describe


@pd.api.extensions.register_dataframe_accessor("clarite_process")
class ClariteProcessDFAccessor(object):
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


@pd.api.extensions.register_dataframe_accessor("clarite_modify")
class ClariteModifyDFAccessor(object):
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        # TODO: Perform any required validation
        if False:
            raise AttributeError("")

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


@pd.api.extensions.register_dataframe_accessor("clarite_plot")
class ClaritePlotDFAccessor(object):
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        # TODO: Perform any required validation
        if False:
            raise AttributeError("")

    def histogram(
        self,
        column: str,
        figsize: Tuple[int, int] = (12, 5),
        title: Optional[str] = None,
        **kwargs,
    ):
        """
        Plot a histogram of the values in the given column.  Takes kwargs for seaborn's distplot.

        Parameters
        ----------
        column: string
            The name of the column that will be plotted
        figsize: tuple(int, int), default (12, 5)
            The figure size of the resulting plot
        title: string or None, default None
            The title used for the plot
        **kwargs:
            Other keyword arguments to pass to the distplot function of Seaborn

        Returns
        -------
        None

        Examples
        --------
        >>> import clarite
        >>> title = f"Discovery: Skew of BMIMBX = {stats.skew(nhanes_discovery_cont['BMXBMI']):.6}"
        >>> nhanes_discovery_cont.clarite_plot.histogram(column="BMXBMI", title=title, bins=100)

        .. image:: _static/plots/histogram.png
        """
        df = self._obj
        plot.histogram(df, column=column, figsize=figsize, **kwargs)

    def distributions(
        self,
        filename: str,
        continuous_kind: str = "count",
        nrows: int = 4,
        ncols: int = 3,
        quality: str = "medium",
        variables: Optional[List[str]] = None,
        sort: bool = True,
    ):
        """
        Create a pdf containing histograms for each binary or categorical variable, and one of several types of plots for each continuous variable.

        Parameters
        ----------
        filename: string
            Name of the saved pdf file.  The extension will be added automatically if it was not included.
        continuous_kind: string
            What kind of plots to use for continuous data.  Binary and Categorical variables will always be shown with histograms.
            One of {'count', 'box', 'violin', 'qq'}
        nrows: int (default=4)
            Number of rows per page
        ncols: int (default=3)
            Number of columns per page
        quality: 'low', 'medium', or 'high'
            Adjusts the DPI of the plots (150, 300, or 1200)
        variables: List[str] or None
            Which variables to plot.  If None, all variables are plotted.
        sort: Boolean (default=True)
            Whether or not to sort variable names

        Returns
        -------
        None

        Examples
        --------
        >>> import clarite
        >>> df[['female', 'occupation', 'LBX074']].clarite_plot.distributions(filename="test")

        .. image:: _static/plots/distributions_count.png

        >>> df[['female', 'occupation', 'LBX074']].clarite_plot.distributions(filename="test", continuous_kind='box')

        .. image:: _static/plots/distributions_box.png

        >>> df[['female', 'occupation', 'LBX074']].clarite_plot.distributions(filename="test", continuous_kind='violin')

        .. image:: _static/plots/distributions_violin.png

        >>> df[['female', 'occupation', 'LBX074']].clarite_plot.distributions(filename="test", continuous_kind='qq')

        .. image:: _static/plots/distributions_qq.png

        """
        df = self._obj
        plot.distributions(df, filename=filename, continuous_kind=continuous_kind,
                           nrows=nrows, ncols=ncols, quality=quality, variables=variables, sort=sort)

    def manhattan(
        self,
        categories: Dict[str, str] = dict(),
        num_labeled: int = 3,
        label_vars: List[str] = list(),
        figsize: Tuple[int, int] = (10, 4),
        dpi: int = 300,
        title: Optional[str] = None,
        colors: List[str] = ["#53868B", "#4D4D4D"],
        background_colors: List[str] = ["#EBEBEB", "#FFFFFF"],
        filename: Optional[str] = None,
    ):
        """
        Create a Manhattan-like plot for EWAS Results

        Parameters
        ----------
        categories: dictionary (string: string)
            A dictionary mapping each variable name to a category name
        num_labeled: int, default 3
            Label the top <num_labeled> results with the variable name
        label_vars: list of strings, default empty list
            Label the named variables
        figsize: tuple(int, int), default (10, 4)
            The figure size of the resulting plot in inches
        dpi: int, default 300
            The figure dots-per-inch
        title: string or None, default None
            The title used for the plot
        colors: List(string, string), default ["#53868B", "#4D4D4D"]
            A list of two colors to use for alternating categories
        background_colors: List(string, string), default ["#EBEBEB", "#FFFFFF"]
            A list of two background colors to use for alternating categories
        filename: Optional str
            If provided, a copy of the plot will be saved to the specified file

        Returns
        -------
        None

        Examples
        --------
        >>> ewas_discovery.clarite_plot.manhattan(categories=data_categories, title="Discovery", filename="discovery.png")

        .. image:: _static/plots/manhattan_single.png
        """
        df = self._obj

        # This is a wrapper around a plotting function which handles plotting multiple datasets
        plot.manhattan(
            dfs={"": df},
            categories=categories,
            num_labeled=num_labeled,
            label_vars=label_vars,
            figsize=figsize,
            dpi=dpi,
            title=title,
            colors=colors,
            background_colors=background_colors,
            filename=filename,
        )


@pd.api.extensions.register_dataframe_accessor("clarite_describe")
class ClariteDescribeDFAccessor(object):
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
