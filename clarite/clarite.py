from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from .ewas import result_columns, corrected_pvalue_columns


@pd.api.extensions.register_dataframe_accessor("clarite")
class ClariteDataframeAccessor(object):
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        # TODO: Perform any required validation
        if False:
            raise AttributeError("")

    def categorize(self, cat_min: int = 3, cat_max: int = 6, cont_min: int = 15):
        """Divide variables into binary, categorical, continuous, and ambiguous dataframes"""
        df = self._obj

        # Double-check parameters
        assert cat_min > 2
        assert cat_min <= cat_max
        assert cont_min > cat_max

        # Create filter series
        num_before = len(df.columns)
        unique_count = df.nunique()

        # No values (All NA)
        zero_filter = unique_count == 0
        num_zero = sum(zero_filter)
        print(f"{num_zero:,} of {num_before:,} variables ({num_zero/num_before:.2%}) had no non-NA values and are discarded.")

        # Single value variables (useless for regression)
        single_filter = unique_count == 1
        num_single = sum(single_filter)
        print(f"{num_single:,} of {num_before:,} variables ({num_single/num_before:.2%}) had only one value and are discarded.")

        # Binary
        binary_filter = unique_count == 2
        num_binary = sum(binary_filter)
        print(f"{num_binary:,} of {num_before:,} variables ({num_binary/num_before:.2%}) are classified as binary (2 values).")
        bin_df = df.loc[:, binary_filter]

        # Categorical
        cat_filter = (unique_count >= cat_min) & (unique_count <= cat_max)
        num_cat = sum(cat_filter)
        print(f"{num_cat:,} of {num_before:,} variables ({num_cat/num_before:.2%}) are classified as categorical ({cat_min} to {cat_max} values).")
        cat_df = df.loc[:, cat_filter]

        # Continuous
        cont_filter = unique_count >= cont_min
        num_cont = sum(cont_filter)
        print(f"{num_cont:,} of {num_before:,} variables ({num_cont/num_before:.2%}) are classified as continuous (>= {cont_min} values).")
        cont_df = df.loc[:, cont_filter]

        # Check
        check_filter = ~zero_filter & ~single_filter & ~binary_filter & ~cat_filter & ~cont_filter
        num_check = sum(check_filter)
        print(f"{num_check:,} of {num_before:,} variables ({num_check/num_before:.2%}) are not classified (between {cat_max} and {cont_min} values).")
        check_df = df.loc[:, check_filter]

        return bin_df, cat_df, cont_df, check_df

    def plot_hist(self, column: str, figsize: Tuple[int, int] = (12, 5), title: Optional[str] = None, **kwargs):
        """Plot a histogram of the values in the given column.  Takes kwargs for seaborn's distplot."""
        df = self._obj
        if title is None:
            title = f"Histogram for {column}"
        if column not in df.columns:
            raise ValueError("'column' must be an existing column in the DataFrame")

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title)
        sns.distplot(df[column], ax=ax, **kwargs)

    def plot_manhattan(self,
                       categories: Dict[str, str] = dict(),
                       num_labeled: int = 3,
                       figsize: Tuple[int, int] = (18, 7),
                       title: Optional[str] = None,
                       colors: List[str] = ["#53868B", "#4D4D4D"],
                       background_colors: List[str] = ["#EBEBEB", "#FFFFFF"]):
        """Create a Manhattan-like plot for EWAS Results"""
        df = self._obj

        if list(df) != result_columns + corrected_pvalue_columns:
            raise ValueError(f"This plot may only be created for EWAS results with corrected p-values added.")

        # Format results
        df = df['pvalue'].to_frame().reset_index()
        df['category'] = df['variable'].apply(lambda v: categories.get(v, "Unknown")).astype('category')
        df['-log10(p_value)'] = -1 * df['pvalue'].apply(np.log10)
        df = df.sort_values(['category', 'variable']).reset_index(drop=True).reset_index()

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        x_labels = []
        x_labels_pos = []
        for num, (name, group) in enumerate(df.groupby('category')):
            # background bars
            ax.axvspan(group['index'].min()-0.5, group['index'].max()+0.5, facecolor=background_colors[num % len(colors)], alpha=0.5)
            # plotted points
            group.plot(kind='scatter', x='index', y='-log10(p_value)', color=colors[num % len(colors)], zorder=2, s=10000/len(df), ax=ax)
            # Record centered position and name of xticks
            x_labels.append(name)
            x_labels_pos.append((group['index'].iloc[-1] - (group['index'].iloc[-1] - group['index'].iloc[0]) / 2))

        # Format plot
        ax.set_xticks(x_labels_pos)
        ax.set_xticklabels(x_labels)
        ax.set_xlim([0, len(df)])
        ax.set_ylim([0, df['-log10(p_value)'].max() + 10])
        ax.set_xlabel('')  # Hide x-axis label since it is obvious
        ax.set_title(title, fontsize=20)
        ax.yaxis.label.set_size(16)

        # Significance line
        significance = -np.log10(0.05/len(df))
        ax.axhline(y=significance, color='red', linestyle='-', zorder=3)
        ax.tick_params(labelrotation=90)
        plt.yticks(fontsize=8)

        # Label top points
        for index, row in df.sort_values('-log10(p_value)', ascending=False).head(n=num_labeled).iterrows():
            ax.text(index+1, row['-log10(p_value)'], str(row['variable']),
                    rotation=0, ha='left', va='center',
                    )
