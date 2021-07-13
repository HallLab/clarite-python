from typing import Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd


def top_results(
    ewas_result: pd.DataFrame,
    pvalue_name: str = "pvalue",
    cutoff: Optional[float] = 0.05,
    num_rows: int = 20,
    figsize: Optional[Tuple[int, int]] = None,
    dpi: int = 300,
    title: Optional[str] = None,
    figure: Optional[plt.figure] = None,
    filename: Optional[str] = None,
):
    """
    Create a dotplot for EWAS Results showing pvalues and beta coefficients

    Parameters
    ----------
    ewas_result: DataFrame
        EWAS Result to plot
    pvalue_name: str
        'pvalue', 'pvalue_fdr', or 'pvalue_bonferroni'
    cutoff: float (default 0.05)
        A vertical line is drawn in the pvalue column to show a significance cutoff
    num_rows: int (default 20)
        How many rows to show in the plot
    figsize: tuple(int, int), default (12, 6)
        The figure size of the resulting plot in inches
    dpi: int, default 300
        The figure dots-per-inch
    title: string or None, default None
        The title used for the plot
    figure: matplotlib Figure or None, default None
        Pass in an existing figure to plot to that instead of creating a new one (ignoring figsize and dpi)
    filename: Optional str
        If provided, a copy of the plot will be saved to the specified file instead of being shown

    Returns
    -------
    None

    Examples
    --------
    >>> clarite.plot.top_results(ewas_result)

    .. image:: ../_static/plot/top_results.png
    """
    # TODO
    # Work with multiple outcomes (subplots)
    # Clearly show what the colors mean
    # Custom colors
    # Error if multiple outcomes are present
    if len(ewas_result.reset_index(drop=False)["Outcome"].unique()) > 1:
        raise ValueError(
            "The 'top_results' plot is limited to displaying results for a single outcome at a time."
        )

    # Ensure corrected pvalues are present
    if pvalue_name == "pvalue_fdr" or pvalue_name == "pvalue_bonferroni":
        if pvalue_name not in list(ewas_result):
            raise ValueError(
                "Missing corrected pvalues in ewas result.  Run clarite.analyze.add_corrected_pvalues"
            )
    elif pvalue_name == "pvalue":
        pass
    else:
        raise ValueError(
            "Incorrect value specified for 'pvalue_name': must be one of 'pvalue', 'pvalue_fdr',"
            " or 'pvalue_bonferroni'."
        )

    # Sort and filter data
    df = (
        ewas_result.sort_values(pvalue_name, ascending=True)
        .head(num_rows)
        .reset_index()
    )

    df["Beta"] = df["Beta"].fillna(0.0)  # Still want to show a point

    # Colors
    type_colors = {
        "binary": "#53868B",  # Has beta (Python)
        "continuous": "#53868B",  # Has beta (Both)
        "categorical": "#4D4D4D",  # No beta (Python)
        "categorical/binary": "#4D4D4D",  # No beta (R)
    }
    palette = (
        df[["Variable", "Variable_type"]].set_index("Variable").squeeze().str.lower()
    )
    palette = palette.to_dict()
    palette = {k: type_colors.get(v, "red") for k, v in palette.items()}

    # Build figure
    sns.set(style="whitegrid")
    # Create Plot and structures to hold category info
    if figure is None:
        figure, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize, dpi=dpi)
    else:
        axes = figure.subplots(nrows=1, ncols=2)

    # Draw vertical lines before plotting points
    if cutoff is not None:
        axes[0].axvline(x=cutoff, ls="--", color="red")  # Significance cutoff
    axes[0].axvline(x=1, ls="-", color="black")  # 1 Pvalue
    axes[1].axvline(x=0, ls="-", color="black")  # 0 Beta

    # Plot points
    sns.stripplot(
        x=pvalue_name,
        y="Variable",
        data=df,
        ax=axes[0],
        size=10,
        orient="h",
        palette=palette,
        linewidth=1,
        edgecolor="w",
        jitter=False,
    )
    sns.stripplot(
        x="Beta",
        y="Variable",
        data=df,
        ax=axes[1],
        size=10,
        orient="h",
        palette=palette,
        linewidth=1,
        edgecolor="w",
        jitter=False,
    )

    # Format
    for ax in axes:
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)

    # Update Axes
    # y-axis labels
    axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0, va="center")
    axes[1].set_yticklabels([])
    axes[1].set_ylabel("")
    # pvalue
    axes[0].set_xscale("log")
    if cutoff is not None:
        xmin = min(df[pvalue_name].min(), cutoff)
        xlabel = f"{pvalue_name} (cutoff = {cutoff:.3f})"
    else:
        xmin = df[pvalue_name].min()
        xlabel = f"{pvalue_name}"
    axes[0].set_xlim(0.1 * xmin, 1)
    axes[0].set_xlabel(xlabel)
    # Beta
    max_beta = df["Beta"].abs().max()
    axes[1].set_xlim(-1.10 * max_beta, 1.1 * max_beta)  # max value +/- 10%

    # Title
    if title is None:
        title = "Top Results"
    figure.suptitle(title, fontsize=20)

    # legend
    legend_elements = []
    if cutoff is not None:
        legend_elements.append(
            Line2D([0], [0], color="red", ls="--", label=f"Pvalue cutoff: {cutoff:.3f}")
        )
    for var_type in list(df["Variable_type"].unique()):
        color = type_colors.get(var_type)
        legend_elements.append(
            Line2D([0], [0], marker="o", color=color, label=var_type, markersize=10)
        )
    axes[0].legend(handles=legend_elements, loc="lower left")

    # Format
    figure.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")
    else:
        plt.show()
