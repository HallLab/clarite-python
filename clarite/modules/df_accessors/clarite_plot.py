from typing import Dict, List, Optional, Tuple

import pandas as pd

from .. import plot


@pd.api.extensions.register_dataframe_accessor("clarite_plot")
class ClaritePlotDFAccessor(object):
    """Available as 'clarite_plot'"""
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