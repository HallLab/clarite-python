from copy import copy
from typing import Dict, List, Optional, Tuple

from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from clarite.modules.analyze import result_columns, corrected_pvalue_columns


def _plot_manhattan(
    dfs: Dict[str, pd.DataFrame],
    pval_column: str = "pvalue",
    categories: Optional[Dict[str, str]] = None,
    cutoffs: Optional[
        List[List[Tuple[str, float, str, str]]]
    ] = None,  # One list of tuples for each df
    num_labeled: int = 3,
    label_vars: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 6),
    dpi: int = 300,
    title: Optional[str] = None,
    figure: Optional[plt.figure] = None,
    colors: List[str] = ["#53868B", "#4D4D4D"],
    background_colors: List[str] = ["#EBEBEB", "#FFFFFF"],
    filename: Optional[str] = None,
    return_figure: bool = False,
):
    """
    Create a manhattan plot.  Easier for the user to expose multiple functions with different specific parameters
    than to use this one big function with all of the parameters.
    """
    # Hardcoded options
    OFFSET = 5  # Spacing between categories

    # Fix optional params
    if categories is None:
        categories = dict()
    if label_vars is None:
        label_vars = []

    # Parameter Validation - EWAS Results
    for df_idx, df in enumerate(dfs.values()):
        missing_cols = set(result_columns + corrected_pvalue_columns) - set(list(df))
        if len(missing_cols) > 0:
            raise ValueError(
                f"This plot may only be created for EWAS results with corrected p-values added. "
                f"DataFrame {df_idx + 1} of {len(dfs)} was missing columns: {', '.join(missing_cols)}"
            )
        if df.index.names != ["Variable", "Outcome"]:
            raise ValueError(
                f"The ewas result dataframes should have an index of ('Variable', 'Outcome')."
                f"DataFrame {df_idx + 1} of {len(dfs)} had '{list(df.index.names)}'"
            )

    # Create a dataframe of pvalues indexed by variable name
    df = (
        pd.DataFrame.from_dict({k: v[pval_column] for k, v in dfs.items()})
        .stack()
        .reset_index()
    )
    df.columns = ("variable", "outcome", "dataset", pval_column)
    df[["variable", "outcome", "dataset"]] = df[
        ["variable", "outcome", "dataset"]
    ].astype("category")

    # Add label str
    if len(df["outcome"].cat.categories) == 1:
        df["label"] = df["variable"]
    else:
        df["label"] = (
            df[["variable", "outcome"]]
            .astype(str)
            .apply(lambda row: f"{row['variable']}, {row['outcome']}", axis=1)
            .astype("category")
        )

    # Add category
    df["category"] = (
        df["variable"].apply(lambda v: categories.get(v, "unknown")).astype("category")
    )

    # Transform pvalues
    log_pval_column = f"-log10({pval_column})"
    df[log_pval_column] = -1 * df[pval_column].apply(np.log10)

    # Sort and extract an 'index' column
    df = df.sort_values(["category", "variable"]).reset_index(drop=True)

    # Update index (actually x position) column by padding between each category
    df["category_x_offset"] = (
        df.groupby("category").ngroup() * OFFSET
    )  # sorted category number, multiplied to pad between categories
    df["xpos"] = (
        df.groupby(["category", "variable"]).ngroup() + df["category_x_offset"]
    )  # sorted category/variable number plus category offset

    # Create Plot and structures to hold category info
    if figure is None:
        figure, axes = plt.subplots(
            len(dfs), 1, figsize=figsize, dpi=dpi, sharex=True, sharey=True
        )
    else:
        axes = figure.subplots(len(dfs), 1, sharex=True, sharey=True)
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
            left = category_data["xpos"].min() - (OFFSET / 2) - 0.5
            right = category_data["xpos"].max() + (OFFSET / 2) + 0.5
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
                        y=log_pval_column,
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
                y=log_pval_column,
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
        # Label top_n points
        labeled_top_n_var_idxs = set()  # Avoid labeling again by variable name later
        if num_labeled > 0:
            for row_idx, row in (
                dataset_data.sort_values(log_pval_column, ascending=False)
                .head(n=num_labeled)
                .iterrows()
            ):
                labeled_top_n_var_idxs.add(row_idx)
                ax.text(
                    row["xpos"] + 1,
                    row[log_pval_column],
                    str(row["label"]),
                    rotation=0,
                    ha="left",
                    va="center",
                )
        # Label points by variable name
        for row_idx, row in dataset_data.loc[
            dataset_data["variable"].isin(label_vars),
        ].iterrows():
            if row_idx in labeled_top_n_var_idxs:
                continue
            ax.text(
                row["xpos"] + 1,
                row[log_pval_column],
                str(row["label"]),
                rotation=0,
                ha="left",
                va="center",
            )

    # Format bottom axes
    ax = axes[-1]
    ax.set_xticks(x_labels_pos)
    ax.set_xticklabels(x_labels)
    ax.tick_params(labelrotation=90)
    ax.set_xlim([df["xpos"].min() - OFFSET, df["xpos"].max() + OFFSET])
    ax.set_ylim([-1, df[log_pval_column].max() + 10])
    ax.set_xlabel("")  # Hide x-axis label since it is obvious

    # Draw cutoffs
    # NOTE: cutoffs values should be in raw pvalue
    if cutoffs is not None:
        if len(cutoffs) != len(dfs):
            raise ValueError(
                "the cutoffs variable must be None or a list of length equal to dfs"
            )
        for df_idx, df_cutoffs in enumerate(cutoffs):
            ax = axes[df_idx]
            for (label, value, color, line_style) in df_cutoffs:
                ax.axhline(
                    y=-np.log10(value),
                    color=color,
                    linestyle=line_style,
                    zorder=3,
                    label=label,
                )
            # Draw Legend
            axes[df_idx].legend(
                loc="upper right", bbox_to_anchor=(1, 1.1), fancybox=True, shadow=True
            )

    # Title
    figure.suptitle(title, fontsize=20)

    # Save
    if return_figure:
        return figure
    else:
        if filename is not None:
            plt.savefig(filename, bbox_inches="tight")
        else:
            plt.show()


def manhattan(
    dfs: Dict[str, pd.DataFrame],
    categories: Optional[Dict[str, str]] = None,
    bonferroni: Optional[float] = 0.05,
    fdr: Optional[float] = None,
    num_labeled: int = 3,
    label_vars: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 6),
    dpi: int = 300,
    title: Optional[str] = None,
    figure: Optional[plt.figure] = None,
    colors: List[str] = ["#53868B", "#4D4D4D"],
    background_colors: List[str] = ["#EBEBEB", "#FFFFFF"],
    filename: Optional[str] = None,
    return_figure: bool = False,
):
    """
    Create a Manhattan-like plot for a list of EWAS Results

    Parameters
    ----------
    dfs: DataFrame
        Dictionary of dataset names to pandas dataframes of ewas results (requires certain columns)
    categories: dictionary (string: string) or None
        A dictionary mapping each variable name to a category name for optional grouping
    bonferroni: float or None (default 0.05)
        Show a cutoff line at the pvalue corresponding to a given bonferroni-corrected pvalue
    fdr: float or None (default None)
        Show a cutoff line at the pvalue corresponding to a given fdr
    num_labeled: int, default 3
        Label the top <num_labeled> results with the variable name
    label_vars: list of strings, default None
        Label the named variables (or pass None to skip labeling this way)
    figsize: tuple(int, int), default (12, 6)
        The figure size of the resulting plot in inches
    dpi: int, default 300
        The figure dots-per-inch
    title: string or None, default None
        The title used for the plot
    figure: matplotlib Figure or None, default None
        Pass in an existing figure to plot to that instead of creating a new one (ignoring figsize and dpi)
    colors: List(string, string), default ["#53868B", "#4D4D4D"]
        A list of colors to use for alternating categories (must be same length as 'background_colors')
    background_colors: List(string, string), default ["#EBEBEB", "#FFFFFF"]
        A list of background colors to use for alternating categories (must be same length as 'colors')
    filename: Optional str
        If provided, a copy of the plot will be saved to the specified file instead of being shown
    return_figure: boolean, default False
        If True, return figure instead of showing or saving the plot. Useful to customize the plot

    Returns
    -------
    figure: matplotlib Figure or None
        If return_figure, returns a matplotlib Figure object. Else returns None

    Examples
    --------
    >>> clarite.plot.manhattan({'discovery':disc_df, 'replication':repl_df}, categories=data_categories, title="EWAS Results")

    .. image:: ../_static/plot/manhattan.png
    """
    # Calculate the significance lines, if any
    cutoffs = []
    for dataset_num, (dataset_name, dataset_data) in enumerate(dfs.items()):
        df_cutoffs = []
        num_tests = dataset_data["pvalue"].count()
        # Bonferroni Line
        if bonferroni is not None:
            bonf_significance = bonferroni / num_tests
            df_cutoffs.append(
                (
                    f"{bonferroni} Bonferroni with {num_tests} tests",
                    bonf_significance,
                    "red",
                    "dashed",
                )
            )
        # FDR Line
        if fdr is not None:
            fdr_cutoff_value = 0
            pvalues = dataset_data.loc[
                ~dataset_data["pvalue"].isna(), "pvalue"
            ].sort_values(ascending=True)
            for i, p in enumerate(pvalues):
                q = ((i + 1) / num_tests) * fdr
                if p < q:
                    fdr_cutoff_value = p
                else:
                    continue
            df_cutoffs.append(
                (f"{fdr} FDR with {num_tests} tests", fdr_cutoff_value, "red", "dotted")
            )
        cutoffs.append(df_cutoffs)

    fig = _plot_manhattan(
        dfs=dfs,
        pval_column="pvalue",
        categories=categories,
        cutoffs=cutoffs,
        num_labeled=num_labeled,
        label_vars=label_vars,
        figsize=figsize,
        dpi=dpi,
        title=title,
        figure=figure,
        colors=colors,
        background_colors=background_colors,
        filename=filename,
        return_figure=return_figure,
    )
    if return_figure:
        return fig


def manhattan_bonferroni(
    dfs: Dict[str, pd.DataFrame],
    categories: Optional[Dict[str, str]] = None,
    cutoff: Optional[float] = 0.05,
    num_labeled: int = 3,
    label_vars: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 6),
    dpi: int = 300,
    title: Optional[str] = None,
    figure: Optional[plt.figure] = None,
    colors: List[str] = ["#53868B", "#4D4D4D"],
    background_colors: List[str] = ["#EBEBEB", "#FFFFFF"],
    filename: Optional[str] = None,
    return_figure: bool = False,
):
    """
    Create a Manhattan-like plot for a list of EWAS Results using Bonferroni significance

    Parameters
    ----------
    dfs: DataFrame
        Dictionary of dataset names to pandas dataframes of ewas results (requires certain columns)
    categories: dictionary (string: string) or None
        A dictionary mapping each variable name to a category name for optional grouping
    cutoff: float or None (default 0.05)
        The pvalue to draw the Bonferroni significance line at (None for no line)
    num_labeled: int, default 3
        Label the top <num_labeled> results with the variable name
    label_vars: list of strings, default None
        Label the named variables (or pass None to skip labeling this way)
    figsize: tuple(int, int), default (12, 6)
        The figure size of the resulting plot in inches
    dpi: int, default 300
        The figure dots-per-inch
    title: string or None, default None
        The title used for the plot
    figure: matplotlib Figure or None, default None
        Pass in an existing figure to plot to that instead of creating a new one (ignoring figsize and dpi)
    colors: List(string, string), default ["#53868B", "#4D4D4D"]
        A list of colors to use for alternating categories (must be same length as 'background_colors')
    background_colors: List(string, string), default ["#EBEBEB", "#FFFFFF"]
        A list of background colors to use for alternating categories (must be same length as 'colors')
    filename: Optional str
        If provided, a copy of the plot will be saved to the specified file instead of being shown
    return_figure: boolean, default False
        If True, return figure instead of showing or saving the plot. Useful to customize the plot

    Returns
    -------
    figure: matplotlib Figure or None
        If return_figure, returns a matplotlib Figure object. Else returns None

    Examples
    --------
    >>> clarite.plot.manhattan_bonferroni({'discovery':disc_df, 'replication':repl_df},
     categories=data_categories, title="EWAS Results")

    .. image:: ../_static/plot/manhattan_bonferroni.png
    """
    # Ensure corrected values are present
    for name, df in dfs.items():
        if "pvalue_bonferroni" not in list(df):
            raise ValueError(
                f"Missing Bonferroni-corrected Pvalues in {name}.  Run clarite.analyze.add_corrected_pvalues"
            )
    # Create cutoff
    if cutoff is not None:
        cutoffs = [[(f"{cutoff} Bonferroni", cutoff, "red", "dashed")] for _ in dfs]
    else:
        cutoffs = None
    fig = _plot_manhattan(
        dfs=dfs,
        pval_column="pvalue_bonferroni",
        categories=categories,
        cutoffs=cutoffs,
        num_labeled=num_labeled,
        label_vars=label_vars,
        figsize=figsize,
        dpi=dpi,
        title=title,
        figure=figure,
        colors=colors,
        background_colors=background_colors,
        filename=filename,
        return_figure=return_figure,
    )
    if return_figure:
        return fig


def manhattan_fdr(
    dfs: Dict[str, pd.DataFrame],
    categories: Optional[Dict[str, str]] = None,
    cutoff: Optional[float] = 0.05,
    num_labeled: int = 3,
    label_vars: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 6),
    dpi: int = 300,
    title: Optional[str] = None,
    figure: Optional[plt.figure] = None,
    colors: List[str] = ["#53868B", "#4D4D4D"],
    background_colors: List[str] = ["#EBEBEB", "#FFFFFF"],
    filename: Optional[str] = None,
    return_figure: bool = False,
):
    """
    Create a Manhattan-like plot for a list of EWAS Results using FDR significance

    Parameters
    ----------
    dfs: DataFrame
        Dictionary of dataset names to pandas dataframes of ewas results (requires certain columns)
    categories: dictionary (string: string) or None
        A dictionary mapping each variable name to a category name for optional grouping
    cutoff: float or None (default 0.05)
        The pvalue to draw the FDR significance line at (None for no line)
    num_labeled: int, default 3
        Label the top <num_labeled> results with the variable name
    label_vars: list of strings, default None
        Label the named variables (or pass None to skip labeling this way)
    figsize: tuple(int, int), default (12, 6)
        The figure size of the resulting plot in inches
    dpi: int, default 300
        The figure dots-per-inch
    title: string or None, default None
        The title used for the plot
    figure: matplotlib Figure or None, default None
        Pass in an existing figure to plot to that instead of creating a new one (ignoring figsize and dpi)
    colors: List(string, string), default ["#53868B", "#4D4D4D"]
        A list of colors to use for alternating categories (must be same length as 'background_colors')
    background_colors: List(string, string), default ["#EBEBEB", "#FFFFFF"]
        A list of background colors to use for alternating categories (must be same length as 'colors')
    filename: Optional str
        If provided, a copy of the plot will be saved to the specified file instead of being shown
    return_figure: boolean, default False
        If True, return figure instead of showing or saving the plot. Useful to customize the plot

    Returns
    -------
    figure: matplotlib Figure or None
        If return_figure, returns a matplotlib Figure object. Else returns None

    Examples
    --------
    >>> clarite.plot.manhattan_fdr({'discovery':disc_df, 'replication':repl_df},
     categories=data_categories, title="EWAS Results")

    .. image:: ../_static/plot/manhattan_fdr.png
    """
    # Ensure corrected values are present
    for name, df in dfs.items():
        if "pvalue_fdr" not in list(df):
            raise ValueError(
                f"Missing FDR-corrected Pvalues in {name}.  Run clarite.analyze.add_corrected_pvalues"
            )
    # Create cutoff
    if cutoff is not None:
        cutoffs = [[(f"{cutoff} FDR", cutoff, "red", "dashed")] for _ in dfs]
    else:
        cutoffs = None
    fig = _plot_manhattan(
        dfs=dfs,
        pval_column="pvalue_fdr",
        categories=categories,
        cutoffs=cutoffs,
        num_labeled=num_labeled,
        label_vars=label_vars,
        figsize=figsize,
        dpi=dpi,
        title=title,
        figure=figure,
        colors=colors,
        background_colors=background_colors,
        filename=filename,
        return_figure=return_figure,
    )
    if return_figure:
        return fig
