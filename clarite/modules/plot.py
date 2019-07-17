"""
Plot
========

Functions that generate plots

  .. autosummary::
     :toctree: modules/plot

     histogram
     distributions
     manhattan
"""


from copy import copy
import datetime
from typing import Dict, List, Optional, Tuple

from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from statsmodels.api import qqplot
import numpy as np
import pandas as pd

from .analyze import result_columns, corrected_pvalue_columns
from .._version import get_versions

clarite_version = get_versions()


def histogram(
    data,
    column: str,
    figsize: Tuple[int, int] = (12, 5),
    title: Optional[str] = None,
    **kwargs,
):
    """
    Plot a histogram of the values in the given column.  Takes kwargs for seaborn's distplot.

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
    **kwargs:
        Other keyword arguments to pass to the distplot function of Seaborn

    Returns
    -------
    None

    Examples
    --------
    >>> import clarite
    >>> title = f"Discovery: Skew of BMIMBX = {stats.skew(nhanes_discovery_cont['BMXBMI']):.6}"
    >>> clarite.plot.histogram(nhanes_discovery_cont, column="BMXBMI", title=title, bins=100)

    .. image:: ../../_static/plot/histogram.png
    """
    if title is None:
        title = f"Histogram for {column}"
    if column not in data.columns:
        raise ValueError("'column' must be an existing column in the DataFrame")

    _, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    sns.distplot(data.loc[~data[column].isna(), column], ax=ax, **kwargs)


def distributions(
    data,
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
    data: pd.DataFrame
        The DataFrame containing data to be plotted
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
    >>> clarite.plot.distributions(df[['female', 'occupation', 'LBX074']], filename="test")

    .. image:: ../../_static/plot/distributions_count.png

    >>> clarite.plot.distributions(df[['female', 'occupation', 'LBX074']], filename="test", continuous_kind='box')

    .. image:: ../../_static/plot/distributions_box.png

    >>> clarite.plot.distributions(df[['female', 'occupation', 'LBX074']], filename="test", continuous_kind='violin')

    .. image:: ../../_static/plot/distributions_violin.png

    >>> clarite.plot.distributions(df[['female', 'occupation', 'LBX074']], filename="test", continuous_kind='qq')

    .. image:: ../../_static/plot/distributions_qq.png

    """
    # Limit variables
    if variables is not None:
        data = data[variables]

    # Check filename
    if not filename.endswith(".pdf"):
        filename += ".pdf"

    # Set DPI
    dpi_dict = {"low": 150, "medium": 300, "high": 1200}
    dpi = dpi_dict.get(quality, None)
    if dpi is None:
        raise ValueError(f"quality was set to '{quality}' which is not a valid value")

    # Make sure file is writeable
    try:
        with PdfPages(filename) as pdf:
            pass
    except OSError:
        raise OSError(f"Unable to write to '{filename}'")

    with PdfPages(filename) as pdf:
        # Determine the number of pages
        plots_per_page = nrows * ncols
        total_pages = (len(data.columns) + (plots_per_page - 1)) // plots_per_page
        print(f"Generating a {total_pages} page PDF for {len(data.columns):,} variables")
        # Starting plot space
        page_num = 1
        row_idx = 0
        col_idx = 0
        # Loop through all variables
        if sort:
            variables = sorted(list(data))
        else:
            variables = list(data)
        for variable in variables:
            if row_idx == 0 and col_idx == 0:
                # New Page
                _ = plt.subplots(squeeze=False, figsize=(8.5, 11), dpi=dpi)
                plt.suptitle(f"Page {page_num}")
            # Plot non-NA values and record the number of those separately (otherwise they can cause issues with generating a KDE)
            ax = plt.subplot2grid((nrows, ncols), (row_idx, col_idx))
            if str(data.dtypes[variable]) == "category":  # binary and categorical
                sns.countplot(data.loc[~data[variable].isna(), variable], ax=ax)
            else:
                if continuous_kind == "count":
                    sns.distplot(
                        data.loc[~data[variable].isna(), variable],
                        kde=False,
                        norm_hist=False,
                        hist_kws={"alpha": 1},
                        ax=ax,
                    )
                elif continuous_kind == "box":
                    sns.boxplot(data.loc[~data[variable].isna(), variable], ax=ax)
                elif continuous_kind == "violin":
                    sns.violinplot(data.loc[~data[variable].isna(), variable], ax=ax)
                elif continuous_kind == "qq":
                    # QQ plots have to be sub-sampled otherwise there are too many points and the pdf is blank
                    d = data.loc[~data[variable].isna(), variable]
                    if len(d) > 400:
                        d = d.sample(n=400, random_state=1)
                    qqplot(d, line="s", fit=True, ax=ax, color="steelblue", alpha=0.7)
                else:
                    raise ValueError(
                        "Unknown value for 'continuous_kind': must be one of {'count', 'box', 'violin', 'qq'}"
                    )
            # Update xlabel with NA information
            na_count = data[variable].isna().sum()
            ax.set_xlabel(
                f"{variable}\n{na_count:,} of {len(data[variable]):,} are NA ({na_count/len(data[variable]):.2%})"
            )
            # Move to next plot space
            col_idx += 1
            if col_idx == ncols:  # Wrap to next row
                col_idx = 0
                row_idx += 1
            if row_idx == nrows:  # Wrap to next page
                row_idx = 0
                page_num += 1
                # Save the current page
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                pdf.savefig()
                plt.close()
        # Save final page, unless a full page was finished and the page_num is now more than total_pages
        if page_num == total_pages:
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            pdf.savefig()
            plt.close()
        # Add metadata
        d = pdf.infodict()
        d["Title"] = "Multipage PDF Example"
        d["Author"] = f"CLARITE {clarite_version}"
        d["Subject"] = "Distribution plots"
        d["CreationDate"] = datetime.datetime.today()
        d["ModDate"] = datetime.datetime.today()


def manhattan(
    dfs: Dict[str, pd.DataFrame],
    categories: Dict[str, str] = dict(),
    num_labeled: int = 3,
    label_vars: List[str] = list(),
    figsize: Tuple[int, int] = (12, 6),
    dpi: int = 300,
    title: Optional[str] = None,
    colors: List[str] = ["#53868B", "#4D4D4D"],
    background_colors: List[str] = ["#EBEBEB", "#FFFFFF"],
    filename: Optional[str] = None,
):
    """
    Create a Manhattan-like plot for a list of EWAS Results

    Parameters
    ----------
    dfs: DataFrame
        Dictionary of dataset names to pandas dataframes of ewas results (requires certain columns)
    categories: dictionary (string: string)
        A dictionary mapping each variable name to a category name
    num_labeled: int, default 3
        Label the top <num_labeled> results with the variable name
    label_vars: list of strings, default empty list
        Label the named variables
    figsize: tuple(int, int), default (12, 6)
        The figure size of the resulting plot in inches
    dpi: int, default 300
        The figure dots-per-inch
    title: string or None, default None
        The title used for the plot
    colors: List(string, string), default ["#53868B", "#4D4D4D"]
        A list of colors to use for alternating categories (must be same length as 'background_colors')
    background_colors: List(string, string), default ["#EBEBEB", "#FFFFFF"]
        A list of background colors to use for alternating categories (must be same length as 'colors')
    filename: Optional str
        If provided, a copy of the plot will be saved to the specified file

    Returns
    -------
    None

    Examples
    --------
    >>> clarite.plot_manhattan({'discovery':disc_df, 'replication':repl_df}, categories=data_categories, title="EWAS Results")

    .. image:: ../../_static/plot/manhattan.png
    """
    # Hardcoded options
    offset = 5  # Spacing between categories

    # Parameter Validation - EWAS Results
    for df_idx, df in enumerate(dfs.values()):
        if list(df) != result_columns + corrected_pvalue_columns:
            raise ValueError(
                f"This plot may only be created for EWAS results with corrected p-values added. "
                f"DataFrame {df_idx + 1} of {len(dfs)} did not have the expected columns."
            )

    # Create a dataframe of pvalues indexed by variable name
    df = (
        pd.DataFrame.from_dict({k: v["pvalue"] for k, v in dfs.items()})
        .stack()
        .reset_index()
    )
    df.columns = ("variable", "phenotype", "dataset", "pvalue")
    df[["variable", "phenotype", "dataset"]] = df[
        ["variable", "phenotype", "dataset"]
    ].astype("category")

    # Add category
    df["category"] = (
        df["variable"].apply(lambda v: categories.get(v, "unknown")).astype("category")
    )

    # Transform pvalues
    df["-log10(p value)"] = -1 * df["pvalue"].apply(np.log10)

    # Sort and extract an 'index' column
    df = df.sort_values(["category", "variable"]).reset_index(drop=True)

    # Update index (actually x position) column by padding between each category
    df["category_x_offset"] = (
        df.groupby("category").ngroup() * offset
    )  # sorted category number, multiplied to pad between categories
    df["xpos"] = (
        df.groupby(["category", "variable"]).ngroup() + df["category_x_offset"]
    )  # sorted category/variable number plus category offset

    # Create Plot and structures to hold category info
    fig, axes = plt.subplots(
        len(dfs), 1, figsize=figsize, dpi=dpi, sharex=True, sharey=True
    )
    x_labels = []
    x_labels_pos = []
    foreground_rectangles = [list() for c in colors]

    # Wrap axes in a list if there is only one plot so that it can be iterated
    if len(dfs) == 1:
        axes = [axes]

    # Plot
    if len(categories) > 0:
        # Include category info
        for category_num, (category_name, category_data) in enumerate(
            df.groupby("category")
        ):
            # background bars
            left = category_data["xpos"].min() - (offset / 2) - 0.5
            right = category_data["xpos"].max() + (offset / 2) + 0.5
            width = right - left
            for ax in axes:
                ax.axvspan(
                    left,
                    right,
                    facecolor=background_colors[category_num % len(colors)],
                    alpha=1,
                )
            # foreground bars
            rect = Rectangle(xy=(left, -1), width=width, height=1)
            foreground_rectangles[category_num % len(colors)].append(rect)
            # plotted points
            for dataset_num, (dataset_name, dataset_data) in enumerate(
                category_data.groupby("dataset")
            ):
                if (
                    len(dataset_data) > 0
                ):  # Sometimes a category has no variables in one dataset
                    dataset_data.plot(
                        kind="scatter",
                        x="xpos",
                        y="-log10(p value)",
                        label=None,
                        color=colors[category_num % len(colors)],
                        zorder=2,
                        s=10000 / len(df),
                        ax=axes[dataset_num],
                    )
            # Record centered position and name of xtick for the category
            x_labels.append(category_name)
            x_labels_pos.append(left + width / 2)

        # Show the foreground rectangles for each category by making a collection then adding it to each axes
        for rect_idx, rect_list in enumerate(foreground_rectangles):
            pc = PatchCollection(
                rect_list,
                facecolor=colors[rect_idx],
                edgecolor=colors[rect_idx],
                alpha=1,
                zorder=2,
            )
            for ax in axes:
                ax.add_collection(
                    copy(pc)
                )  # have to add a copy since the same object can't be added to multiple axes
    else:
        # Just plot variables
        for dataset_num, (dataset_name, dataset_data) in enumerate(
            df.groupby("dataset")
        ):
            # Plot points
            dataset_data.plot(
                kind="scatter",
                x="xpos",
                y="-log10(p value)",
                label=None,
                color=colors[0],
                zorder=2,
                s=10000 / len(df),
                ax=axes[dataset_num],
            )
        # Plot a single rectangle on each axis
        for ax in axes:
            rect = Rectangle(xy=(0, -1), width=df["xpos"].max(), height=1)
            pc = PatchCollection(
                [rect], facecolor=colors[0], edgecolor=colors[0], alpha=1, zorder=2
            )
            ax.add_collection(pc)

    # Format plot
    for dataset_num, (dataset_name, dataset_data) in enumerate(df.groupby("dataset")):
        ax = axes[dataset_num]
        # Set title
        ax.set_title(dataset_name, fontsize=14)
        ax.yaxis.label.set_size(16)
        # Update y-axis
        plt.yticks(fontsize=8)
        # Label points
        top_n = dataset_data.sort_values("-log10(p value)", ascending=False).head(
            n=num_labeled
        )["variable"]
        labeled = set(label_vars) | set(top_n)
        for _, row in dataset_data.loc[
            dataset_data["variable"].isin(labeled),
        ].iterrows():
            ax.text(
                row["xpos"] + 1,
                row["-log10(p value)"],
                str(row["variable"]),
                rotation=0,
                ha="left",
                va="center",
            )

    # Format bottom axes
    ax = axes[-1]
    ax.set_xticks(x_labels_pos)
    ax.set_xticklabels(x_labels)
    ax.tick_params(labelrotation=90)
    ax.set_xlim([df["xpos"].min() - offset, df["xpos"].max() + offset])
    ax.set_ylim([-1, df["-log10(p value)"].max() + 10])
    ax.set_xlabel("")  # Hide x-axis label since it is obvious

    # Significance line for each dataset
    for dataset_num, (dataset_name, dataset_data) in enumerate(df.groupby("dataset")):
        num_tests = dataset_data["pvalue"].count()
        significance = -np.log10(0.05 / num_tests)
        axes[dataset_num].axhline(
            y=significance,
            color="red",
            linestyle="-",
            zorder=3,
            label=f"0.05 Bonferroni with {num_tests} tests",
        )
        axes[dataset_num].legend(
            loc="upper right", bbox_to_anchor=(1, 1.1), fancybox=True, shadow=True
        )

    # Title
    fig.suptitle(title, fontsize=20)

    # Save
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")
    plt.show()
