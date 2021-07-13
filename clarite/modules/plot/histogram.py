from typing import Optional, Tuple


import matplotlib.pyplot as plt
import seaborn as sns

from clarite.internal.utilities import _get_dtypes


def histogram(
    data,
    column: str,
    figsize: Tuple[int, int] = (12, 5),
    title: Optional[str] = None,
    figure: Optional[plt.figure] = None,
    **kwargs,
):
    """
    Plot a histogram of the values in the given column.

    Parameters
    ----------
    data: pd.DataFrame
        The DataFrame containing data to be plotted
    column: string
        The name of the column that will be plotted
    figsize: tuple(int, int), default (12, 5)
        The figure size of the resulting plot
    title: string or None, default None
        The title used for the plot
    figure: matplotlib Figure or None, default None
        Pass in an existing figure to plot to that instead of creating a new one (ignoring figsize)
    **kwargs:
        Other keyword arguments to pass to the histplot or catplot function of Seaborn

    Returns
    -------
    None

    Examples
    --------
    >>> import clarite
    >>> title = f"Discovery: Skew of BMIMBX = {stats.skew(nhanes_discovery_cont['BMXBMI']):.6}"
    >>> clarite.plot.histogram(nhanes_discovery_cont, column="BMXBMI", title=title, bins=100)

    .. image:: ../_static/plot/histogram.png
    """
    if title is None:
        title = f"Histogram for {column}"
    if column not in data.columns:
        raise ValueError("'column' must be an existing column in the DataFrame")

    if figure is None:
        _, ax = plt.subplots(figsize=figsize)
    else:
        ax = figure.subplots()
    ax.set_title(title)
    # Determine type, which determines which plot function to use
    print(kwargs)
    datatype = _get_dtypes(data[column])[column]
    if datatype == "continuous":
        sns.histplot(x=data.loc[~data[column].isna(), column], ax=ax, **kwargs)
    elif datatype == "categorical":
        sns.countplot(x=data[column], ax=ax, **kwargs)
    else:
        raise ValueError(f"Can't plot a histogram with data of type {datatype}.")
