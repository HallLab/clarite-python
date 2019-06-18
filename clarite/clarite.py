from typing import Dict, List, Optional, Tuple
import datetime

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import numpy as np
import pandas as pd
from statsmodels.api import qqplot

from .ewas import result_columns, corrected_pvalue_columns
from .utilities import _validate_skip_only
from ._version import get_versions

clarite_version = get_versions()

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
        >>> nhanes_bin, nhanes_cat, nhanes_cont, nhanes_other = nhanes.clarite.categorize()
        10 of 945 variables (1.06%) had no non-NA values and are discarded.
        33 of 945 variables (3.49%) had only one value and are discarded.
        361 of 945 variables (38.20%) are classified as binary (2 values).
        44 of 945 variables (4.66%) are classified as categorical (3 to 6 values).
        461 of 945 variables (48.78%) are classified as continuous (>= 15 values).
        36 of 945 variables (3.81%) are not classified (between 6 and 15 values).
        """
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

        # Other
        other_filter = ~zero_filter & ~single_filter & ~binary_filter & ~cat_filter & ~cont_filter
        num_other = sum(other_filter)
        print(f"{num_other:,} of {num_before:,} variables ({num_other/num_before:.2%}) are not classified (between {cat_max} and {cont_min} values).")
        other_df = df.loc[:, other_filter]

        return bin_df, cat_df, cont_df, other_df

    ##################
    # Column Filters #
    ##################

    def colfilter_percent_zero(self, proportion: float = 0.9, skip: Optional[List[str]] = None, only: Optional[List[str]] = None):
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
        >>> nhanes_discovery_cont = nhanes_discovery_cont.clarite.colfilter_percent_zero()
        Removed 30 of 369 variables (8.13%) which were equal to zero in at least 90.00% of non-NA observations.
        """
        df = self._obj
        columns = _validate_skip_only(list(df), skip, only)
        num_before = len(df.columns)

        percent_value = df.apply(lambda col: sum(col == 0) / col.count())
        kept = (percent_value < proportion) | ~df.columns.isin(columns)
        num_removed = num_before - sum(kept)

        print(f"Removed {num_removed:,} of {num_before:,} variables ({num_removed/num_before:.2%}) "
              f"which were equal to zero in at least {proportion:.2%} of non-NA observations.")
        return df.loc[:, kept]

    def colfilter_min_n(self, n: int = 200, skip: Optional[List[str]] = None, only: Optional[List[str]] = None):
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
        >>> nhanes_discovery_bin = nhanes_discovery_bin.clarite.colfilter_min_n()
        Removed 129 of 361 variables (35.73%) which had less than 200 values
        """
        df = self._obj
        columns = _validate_skip_only(list(df), skip, only)
        num_before = len(df.columns)

        counts = df.count()  # by default axis=0 (rows) so counts number of non-NA rows in each column
        kept = (counts >= n) | ~df.columns.isin(columns)
        num_removed = num_before - sum(kept)

        print(f"Removed {num_removed:,} of {num_before:,} variables ({num_removed/num_before:.2%}) which had less than {n} values")
        return df.loc[:, kept]

    def colfilter_min_cat_n(self, n: int = 200, skip: Optional[List[str]] = None, only: Optional[List[str]] = None):
        """
        Remove columns which have less than <n> unique values in each category

        Parameters
        ----------
        n: int, default 200
            The minimum number of unique values required in each category in order for a variable not to be filtered
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
        >>> nhanes_discovery_bin = nhanes_discovery_bin.clarite.colfilter_min_cat_n()
        Removed 159 of 232 variables (68.53%) which had a category with less than 200 values
        """
        df = self._obj
        columns = _validate_skip_only(list(df), skip, only)
        num_before = len(df.columns)

        min_category_counts = df.apply(lambda col: col.value_counts().min())
        kept = (min_category_counts >= n) | ~df.columns.isin(columns)
        num_removed = num_before - sum(kept)

        print(f"Removed {num_removed:,} of {num_before:,} variables ({num_removed/num_before:.2%}) which had a category with less than {n} values")
        return df.loc[:, kept]

    ###############
    # Row Filters #
    ###############

    def rowfilter_incomplete_observations(self, columns):
        """
        Remove rows when any of the passed columns has a null value

        Parameters
        ----------
        columns: list
            The columns that may not contain null values

        Returns
        -------
        df: pd.DataFrame
            The filtered DataFrame

        Examples
        --------
        >>> nhanes = nhanes.clarite.rowfilter_incomplete_observations([phenotype] + covariates)
        Removed 3,687 of 22,624 rows (16.30%) due to NA values in the specified columns
        """
        invalid_names = set(columns) - set(list(self._obj))
        if len(invalid_names) > 0:
            raise ValueError(f"Invalid column names were provided: {', '.join(invalid_names)}")

        keep_IDs = self._obj[columns].isnull().sum(axis=1) == 0  # Number of NA in each row is 0
        n_removed = len(self._obj) - sum(keep_IDs)

        print(f"Removed {n_removed:,} of {len(self._obj):,} rows ({n_removed/len(self._obj):.2%}) due to NA values in the specified columns")
        return self._obj[keep_IDs]

    ############
    # Plotting #
    ############
    def plot_hist(self, column: str, figsize: Tuple[int, int] = (12, 5), title: Optional[str] = None, **kwargs):
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
        >>> title = f"Discovery: Skew of BMIMBX = {stats.skew(nhanes_discovery_cont['BMXBMI']):.6}"
        >>> nhanes_discovery_cont.clarite.plot_hist(column="BMXBMI", title=title, bins=100)

        .. image:: _static/plots/plot_hist.png
        """
        df = self._obj
        if title is None:
            title = f"Histogram for {column}"
        if column not in df.columns:
            raise ValueError("'column' must be an existing column in the DataFrame")

        _, ax = plt.subplots(figsize=figsize)
        ax.set_title(title)
        sns.distplot(df[column], ax=ax, **kwargs)

    def plot_distributions(self,
                           filename: str,
                           continuous_kind: str = 'count',
                           nrows: int = 4,
                           ncols: int = 3,
                           quality: str = "medium",
                           variables: Optional[List[str]] = None,
                           sort: bool = True):
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
        >>> df[['female', 'occupation', 'LBX074']].clarite.plot_distributions(filename="test")

        .. image:: _static/plots/plot_distributions_count.png

        >>> df[['female', 'occupation', 'LBX074']].clarite.plot_distributions(filename="test", continuous_kind='box')

        .. image:: _static/plots/plot_distributions_box.png

        >>> df[['female', 'occupation', 'LBX074']].clarite.plot_distributions(filename="test", continuous_kind='violin')

        .. image:: _static/plots/plot_distributions_violin.png

        >>> df[['female', 'occupation', 'LBX074']].clarite.plot_distributions(filename="test", continuous_kind='qq')

        .. image:: _static/plots/plot_distributions_qq.png

        """
        df = self._obj

        # Limit variables
        if variables is not None:
            df = df[variables]

        # Check filename
        if not filename.endswith(".pdf"):
            filename += ".pdf"

        # Set DPI
        dpi_dict = {'low':150, 'medium':300, 'high':1200}
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
            total_pages = (len(df.columns) + (plots_per_page - 1))//plots_per_page
            print(f"Generating a {total_pages} page PDF for {len(df.columns):,} variables")
            # Starting plot space
            page_num = 1
            row_idx = 0
            col_idx = 0
            # Loop through all variables
            if sort:
                variables = sorted(list(df))
            else:
                variables = list(df)
            for variable in variables:
                if row_idx == 0 and col_idx == 0:
                    # New Page
                    _ = plt.subplots(squeeze=False, figsize=(8.5, 11), dpi=dpi)
                    plt.suptitle(f"Page {page_num}")
                # Plot non-NA values and record the number of those separately (otherwise they can cause issues with generating a KDE)
                ax = plt.subplot2grid((nrows, ncols), (row_idx, col_idx))
                if str(df.dtypes[variable]) == 'category':
                    sns.countplot(df.loc[~df[variable].isna(), variable], ax=ax)
                else:
                    if continuous_kind == 'count':
                        sns.distplot(df.loc[~df[variable].isna(), variable], kde=False, norm_hist=False, hist_kws={'alpha':1}, ax=ax)
                    elif continuous_kind == 'box':
                        sns.boxplot(df.loc[~df[variable].isna(), variable], ax=ax)
                    elif continuous_kind == 'violin':
                        sns.violinplot(df.loc[~df[variable].isna(), variable], ax=ax)
                    elif continuous_kind == 'qq':
                        # QQ plots have to be sub-sampled otherwise there are too many points and the pdf is blank
                        d = df.loc[~df[variable].isna(), variable]
                        if len(d) > 400:
                            d = d.sample(n=400, random_state=1)
                        qqplot(d, line='s', fit=True, ax=ax, color='steelblue', alpha=0.7)
                    else:
                        raise ValueError("Unknown value for 'continuous_kind': must be one of {'count', 'box', 'violin', 'qq'}")
                # Update xlabel with NA information
                na_count = df[variable].isna().sum()
                ax.set_xlabel(f"{variable}\n{na_count:,} of {len(df[variable]):,} are NA ({na_count/len(df[variable]):.2%})")
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
            d['Title'] = 'Multipage PDF Example'
            d['Author'] = f"CLARITE {clarite_version}"
            d['Subject'] = 'Distribution plots'
            d['CreationDate'] = datetime.datetime.today()
            d['ModDate'] = datetime.datetime.today()
    

    def plot_manhattan(self,
                       categories: Dict[str, str] = dict(),
                       num_labeled: int = 3,
                       figsize: Tuple[int, int] = (18, 7),
                       title: Optional[str] = None,
                       colors: List[str] = ["#53868B", "#4D4D4D"],
                       background_colors: List[str] = ["#EBEBEB", "#FFFFFF"]):
        """
        Create a Manhattan-like plot for EWAS Results

        Parameters
        ----------
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
        df = self._obj

        if list(df) != result_columns + corrected_pvalue_columns:
            raise ValueError(f"This plot may only be created for EWAS results with corrected p-values added.")

        # Format results
        df = df['pvalue'].to_frame().reset_index()
        df['category'] = df['variable'].apply(lambda v: categories.get(v, "Unknown")).astype('category')
        df['-log10(p_value)'] = -1 * df['pvalue'].apply(np.log10)
        df = df.sort_values(['category', 'variable']).reset_index(drop=True).reset_index()

        # Plot
        _, ax = plt.subplots(1, 1, figsize=figsize)

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

    ######################################
    # Exploratory Stats and Calculations #
    ######################################
    def get_correlations(self, threshold: float = 0.75):
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
        >>> correlations = df.clarite.get_correlations(threshold=0.9)
        >>> correlations.head()
                           var1      var2  correlation
        36704  supplement_count  DSDCOUNT     1.000000
        32807          DR1TM181  DR1TMFAT     0.997900
        33509          DR1TP182  DR1TPFAT     0.996172
        39575          DRD370FQ  DRD370UQ     0.987974
        35290          DR1TS160  DR1TSFAT     0.984733
        """
        df = self._obj
        # Get correlaton matrix
        correlation = df.corr()
        # Keep only the upper triangle to avoid listing both a-b and b-a correlations
        correlation = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(np.bool))
        # Stack and rename into the desired format
        correlation = correlation.stack().rename('correlation').rename_axis(['var1', 'var2']).reset_index()
        # Remove those with correlation below threshold
        correlation = correlation.loc[correlation['correlation'].abs() >= threshold, ]
        # Sort by absolute value and return
        return correlation.reindex(correlation['correlation'].abs().sort_values(ascending=False).index)


    def get_freq_table(self):
        """
        Return the count of each unique value for all categorical variables.  Non-categorical typed variables
        will return a single row with a value of '<Non-Categorical Values>' and the number of non-NA values.

        Returns
        -------
        result: pd.DataFrame
            DataFrame listing variable, value, and count for each categorical variable

        Examples
        --------
        >>> df.clarite.get_freq_table().head(n=10)
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

        # Define a function to be applied to each categorical variable
        def formatted_value_counts(var_name: str, df: pd.DataFrame):
            if str(df[var_name].dtype)=='category':
                df = df[var_name].value_counts().reset_index().rename({'index':'value', var_name:'count'}, axis='columns')
                df['variable'] = var_name
                return df[['variable', 'value', 'count']]  # reorder columns
            else:
                return pd.DataFrame.from_dict({'variable':[var_name],
                                               'value':['<Non-Categorical Values>'],
                                               'count':[df[var_name].count()]})

        return pd.concat([formatted_value_counts(var_name, df) for var_name in list(df)]).reset_index(drop=True)


    def get_percent_na(self):
        """
        Return the percent of observations that are NA for each variable

        Returns
        -------
        result: pd.Series
            Series listing percent NA for each variable

        Examples
        --------
        >>> df.clarite.get_percent_na()
        SDDSRVYR                 0.000000
        female                   0.000000
        LBXHBC                   0.049321
        LBXHBS                   0.049873
        """
        df = self._obj
        return 1 - (df.count() / df.apply(len))