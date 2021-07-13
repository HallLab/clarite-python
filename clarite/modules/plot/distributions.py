try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

import datetime
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from statsmodels.api import qqplot
import click

clarite_version = importlib_metadata.version("clarite")


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
    filename: string or pathlib.Path
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

    .. image:: ../_static/plot/distributions_count.png

    >>> clarite.plot.distributions(df[['female', 'occupation', 'LBX074']], filename="test", continuous_kind='box')

    .. image:: ../_static/plot/distributions_box.png

    >>> clarite.plot.distributions(df[['female', 'occupation', 'LBX074']], filename="test", continuous_kind='violin')

    .. image:: ../_static/plot/distributions_violin.png

    >>> clarite.plot.distributions(df[['female', 'occupation', 'LBX074']], filename="test", continuous_kind='qq')

    .. image:: ../_static/plot/distributions_qq.png

    """
    # Limit variables
    if variables is not None:
        data = data[variables]

    # Check filename, adding ".pdf" if needed
    if type(filename) == str:
        filename = Path(filename)
    if filename.suffix != "pdf":
        filename = Path(str(filename) + ".pdf")

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
        click.echo(
            f"Generating a {total_pages} page PDF for {len(data.columns):,} variables"
        )
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
                sns.countplot(x=data.loc[~data[variable].isna(), variable], ax=ax)
            else:
                if continuous_kind == "count":
                    sns.distplot(
                        x=data.loc[~data[variable].isna(), variable],
                        kde=False,
                        norm_hist=False,
                        hist_kws={"alpha": 1},
                        ax=ax,
                    )
                elif continuous_kind == "box":
                    sns.boxplot(x=data.loc[~data[variable].isna(), variable], ax=ax)
                elif continuous_kind == "violin":
                    sns.violinplot(x=data.loc[~data[variable].isna(), variable], ax=ax)
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
                f"{variable}\n{na_count:,} of {len(data[variable]):,} are NA ({na_count / len(data[variable]):.2%})"
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
