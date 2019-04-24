from typing import Optional, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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
