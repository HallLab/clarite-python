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
                   figsize: Tuple[int, int] = (18, 7),
                   title: Optional[str] = None,
                   colors: List[str] = ["#53868B", "#4D4D4D"],
                   background_colors: List[str] = ["#EBEBEB", "#FFFFFF"]):
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
    figsize: tuple(int, int), default (12, 5)
        The figure size of the resulting plot
    title: string or None, default None
        The title used for the plot
    colors: List(string, string), default ["#53868B", "#4D4D4D"]
        A list of two colors to use for alternating categories
    background_colors: List(string, string), default ["#EBEBEB", "#FFFFFF"]
        A list of two background colors to use for alternating categories

    Returns
    -------
    None

    Examples
    --------
    >>> ewas_replication.clarite.plot_manhattan(categories=data_categories, title="Replication")

    .. image:: _static/plots/plot_manhattan.png
    """

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
    df = df.sort_values(['category', 'variable']).reset_index(drop=True).reset_index()

    # Update index (actually x position) column by padding between each category
    offset = 5
    df['category_x_offset'] = df.groupby('category').ngroup() * offset
    df['index'] += df['category_x_offset']

    # Plot
    _, ax = plt.subplots(1, 1, figsize=figsize)

    x_labels = []
    x_labels_pos = []
    foreground_rectangles = [list() for c in colors]
    for category_num, (category_name, category_data) in enumerate(df.groupby('category')):
        # background bars
        left = category_data['index'].min() - (offset / 2) - 0.5
        right = category_data['index'].max() + (offset / 2) + 0.5
        width = right - left
        ax.axvspan(left, right, facecolor=background_colors[category_num % len(colors)], alpha=0.5)
        # foreground bars
        rect = Rectangle(xy=(right, 0), width=width, height=1, angle=180.0)
        foreground_rectangles[category_num % len(colors)].append(rect)
        ax.axvspan(left, right, facecolor=background_colors[category_num % len(colors)], alpha=0.5, )
        # plotted points
        category_data.plot(kind='scatter', x='index', y='-log10(p value)',
                           color=colors[category_num % len(colors)], zorder=2, s=10000/len(df), ax=ax)
        # Record centered position and name of xticks
        x_labels.append(category_name)
        x_labels_pos.append(left + width/2)

    # Show the foreground rectangles by making a collection then adding it
    for rect_idx, rect_list in enumerate(foreground_rectangles):
        pc = PatchCollection(rect_list, facecolor=colors[rect_idx], edgecolor=colors[rect_idx], alpha=1, zorder=2)
        ax.add_collection(pc)

    # Format plot
    ax.set_xticks(x_labels_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlim([df['index'].min() - offset, df['index'].max() + offset])
    ax.set_ylim([-1, df['-log10(p value)'].max() + 10])
    ax.set_xlabel('')  # Hide x-axis label since it is obvious
    ax.set_title(title, fontsize=20)
    ax.yaxis.label.set_size(16)
    ax.tick_params(labelrotation=90)
    plt.yticks(fontsize=8)

    # Significance line
    significance = -np.log10(0.05/len(df))
    ax.axhline(y=significance, color='red', linestyle='-', zorder=3)

    # Label top points
    for index, row in df.sort_values('-log10(p value)', ascending=False).head(n=num_labeled).iterrows():
        ax.text(index + 1 + row['category_x_offset'], row['-log10(p value)'], str(row['variable']),
                rotation=0, ha='left', va='center',
                )
