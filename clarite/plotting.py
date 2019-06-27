from copy import copy
from typing import Dict, List, Optional, Tuple

from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from .ewas import result_columns, corrected_pvalue_columns


def plot_manhattan(dfs: Dict[str, pd.DataFrame],
                   categories: Dict[str, str] = dict(),
                   num_labeled: int = 3,
                   label_vars: List[str] = list(),
                   figsize: Tuple[int, int] = (18, 7),
                   title: Optional[str] = None,
                   colors: List[str] = ["#53868B", "#4D4D4D"],
                   background_colors: List[str] = ["#EBEBEB", "#FFFFFF"],
                   filename: Optional[str] = None):
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
    figsize: tuple(int, int), default (12, 5)
        The figure size of the resulting plot
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

    .. image:: _static/plots/plot_manhattan.png
    """
    # Hardcoded options
    offset = 5  # Spacing between categories

    # Parameter Validation - EWAS Results
    for df_idx, df in enumerate(dfs.values()):
        if list(df) != result_columns + corrected_pvalue_columns:
            raise ValueError(f"This plot may only be created for EWAS results with corrected p-values added. "
                             f"DataFrame {df_idx} of {len(dfs)} did not have the expected columns.")

    # Create a dataframe of pvalues indexed by variable name
    df = pd.DataFrame.from_dict({k: v['pvalue'] for k, v in dfs.items()}).stack().reset_index()
    df.columns = ('variable', 'phenotype', 'dataset', 'pvalue')
    df[['variable', 'phenotype', 'dataset']] = df[['variable', 'phenotype', 'dataset']] .astype('category')

    # Add category
    df['category'] = df['variable'].apply(lambda v: categories.get(v, "unknown")).astype('category')

    # Transform pvalues
    df['-log10(p value)'] = -1 * df['pvalue'].apply(np.log10)

    # Sort and extract an 'index' column
    df = df.sort_values(['category', 'variable']).reset_index(drop=True)

    # Update index (actually x position) column by padding between each category
    df['category_x_offset'] = df.groupby('category').ngroup() * offset  # sorted category number, multiplied to pad between categories
    df['xpos'] = df.groupby(['category', 'variable']).ngroup() + df['category_x_offset']  # sorted category/variable number plus category offset

    # Create Plot and structures to hold category info
    fig, axes = plt.subplots(len(dfs), 1, figsize=figsize, sharex=True, sharey=True)
    x_labels = []
    x_labels_pos = []
    foreground_rectangles = [list() for c in colors]

    # Wrap axes in a list if there is only one plot so that it can be iterated
    if len(dfs) == 1:
        axes = [axes]

    # Plot
    if len(categories) > 0:
        # Include category info
        for category_num, (category_name, category_data) in enumerate(df.groupby('category')):
            # background bars
            left = category_data['xpos'].min() - (offset / 2) - 0.5
            right = category_data['xpos'].max() + (offset / 2) + 0.5
            width = right - left
            for ax in axes:
                ax.axvspan(left, right, facecolor=background_colors[category_num % len(colors)], alpha=1)
            # foreground bars
            rect = Rectangle(xy=(left, -1), width=width, height=1)
            foreground_rectangles[category_num % len(colors)].append(rect)
            # plotted points
            for dataset_num, (dataset_name, dataset_data) in enumerate(category_data.groupby('dataset')):
                if len(dataset_data) > 0:  # Sometimes a category has no variables in one dataset
                    dataset_data.plot(kind='scatter', x='xpos', y='-log10(p value)', label=None,
                                      color=colors[category_num % len(colors)], zorder=2, s=10000/len(df), ax=axes[dataset_num])
            # Record centered position and name of xtick for the category
            x_labels.append(category_name)
            x_labels_pos.append(left + width/2)

        # Show the foreground rectangles for each category by making a collection then adding it to each axes
        for rect_idx, rect_list in enumerate(foreground_rectangles):
            pc = PatchCollection(rect_list, facecolor=colors[rect_idx], edgecolor=colors[rect_idx], alpha=1, zorder=2)
            for ax in axes:
                ax.add_collection(copy(pc))  # have to add a copy since the same object can't be added to multiple axes
    else:
        # Just plot variables
        for dataset_num, (dataset_name, dataset_data) in enumerate(df.groupby('dataset')):
            # Plot points
            dataset_data.plot(kind='scatter', x='xpos', y='-log10(p value)', label=None,
                              color=colors[0], zorder=2, s=10000/len(df), ax=axes[dataset_num])
        # Plot a single rectangle on each axis
        for ax in axes:
            rect = Rectangle(xy=(0, -1), width=df['xpos'].max(), height=1)
            pc = PatchCollection([rect], facecolor=colors[0], edgecolor=colors[0], alpha=1, zorder=2)
            ax.add_collection(pc)

    # Format plot
    for dataset_num, (dataset_name, dataset_data) in enumerate(df.groupby('dataset')):
        ax = axes[dataset_num]
        # Set title
        ax.set_title(dataset_name, fontsize=14)
        ax.yaxis.label.set_size(16)
        # Update y-axis
        plt.yticks(fontsize=8)
        # Label points
        top_n = dataset_data.sort_values('-log10(p value)', ascending=False).head(n=num_labeled)['variable']
        labeled = set(label_vars) | set(top_n)
        for _, row in dataset_data.loc[dataset_data['variable'].isin(labeled), ].iterrows():
            ax.text(row['xpos'] + 1, row['-log10(p value)'], str(row['variable']),
                    rotation=0, ha='left', va='center')

    # Format bottom axes
    ax = axes[-1]
    ax.set_xticks(x_labels_pos)
    ax.set_xticklabels(x_labels)
    ax.tick_params(labelrotation=90)
    ax.set_xlim([df['xpos'].min() - offset, df['xpos'].max() + offset])
    ax.set_ylim([-1, df['-log10(p value)'].max() + 10])
    ax.set_xlabel('')  # Hide x-axis label since it is obvious

    # Significance line for each dataset
    for dataset_num, (dataset_name, dataset_data) in enumerate(df.groupby('dataset')):
        num_tests = dataset_data['pvalue'].count()
        significance = -np.log10(0.05/num_tests)
        axes[dataset_num].axhline(y=significance, color='red', linestyle='-', zorder=3, label=f"0.05 Bonferroni with {num_tests} tests")
        axes[dataset_num].legend(loc='upper right', bbox_to_anchor=(1, 1.1), fancybox=True, shadow=True)

    # Title
    fig.suptitle(title, fontsize=20)

    # Save
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")
    plt.show()
