from typing import Optional, List, Union

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


def add_corrected_pvalues(
    data: pd.DataFrame,
    pvalue: str = "pvalue",
    groupby: Optional[Union[str, List[str]]] = None,
):
    """
    Calculate bonferroni and FDR pvalues and sort by increasing FDR (in-place).
    Rows with a missing pvalue are not counted as a test.

    Parameters
    ----------
    data:
        A dataframe that will be modified in-place to add corrected pvalues
    pvalue:
        Name of a column in data that the calculations will be based on.
    groupby:
        A name or list of names of columns (including index columns) that will be used to group rows before performing
        calculations. This is meant to be used when multiple rows are present with repeated pvalues based on the same
        test. This will reduce the number of tests.  For example, grouping by ["Term1", "Term2"] in interaction results
        to apply corrections to the LRT_pvalue when betas are reported (which creates more rows than the number of
        tests)

    Returns
    -------
    None

    Examples
    --------
    >>> clarite.analyze.add_corrected_pvalues(ewas_discovery)

    >>> clarite.analyze.add_corrected_pvalues(interaction_result, pvalue='Beta_pvalue')

    >>> clarite.analyze.add_corrected_pvalues(interaction_result, pvalue='LRT_pvalue', groupby=["Term1", "Term2"])
    """
    # Test specifications
    if pvalue not in data.columns:
        raise ValueError(f"'{pvalue}' is not a column in the passed data")
    if groupby is not None:
        if type(groupby) == str:
            if (groupby not in data.columns) and (groupby not in data.index.names):
                raise ValueError(f"'{groupby}' is not a column in the passed data")
        elif type(groupby) == list:
            for g in groupby:
                if (g not in data.columns) and (g not in data.index.names):
                    raise ValueError(f"'{g}' is not a column in the passed data")

    # NA by default
    bonf_name = f"{pvalue}_bonferroni"
    fdr_name = f"{pvalue}_fdr"

    if ~(data[pvalue].isna()).sum() == 0:
        # Return with NA results if there are no pvalues
        data[bonf_name] = np.nan
        data[fdr_name] = np.nan
        return
    elif groupby is None:
        # Start with NA values
        data[bonf_name] = np.nan
        data[fdr_name] = np.nan
        # Calculate values, ignoring NA pvalues
        data.loc[~data[pvalue].isna(), bonf_name] = multipletests(
            data.loc[~data[pvalue].isna(), pvalue], method="bonferroni"
        )[1]
        data.loc[~data[pvalue].isna(), fdr_name] = multipletests(
            data.loc[~data[pvalue].isna(), pvalue], method="fdr_bh"
        )[1]
        # Sort
        data.sort_values(by=[fdr_name, bonf_name], inplace=True)
    elif groupby is not None:
        # Reset index in case groupby columns are in the index
        index_cols = data.index.names
        data.reset_index(drop=False, inplace=True)

        # Get the first value from each
        first = ~data.duplicated(subset=groupby, keep="first")
        bonf_result = pd.Series(
            multipletests(
                data.loc[first & ~data[pvalue].isna(), pvalue], method="bonferroni"
            )[1],
            index=data[first][groupby],
        ).to_dict()
        fdr_result = pd.Series(
            multipletests(
                data.loc[first & ~data[pvalue].isna(), pvalue], method="fdr_bh"
            )[1],
            index=data[first][groupby],
        ).to_dict()
        # Expand results to duplicated rows
        data[bonf_name] = data[groupby].apply(
            lambda g: bonf_result.get(g, np.nan)
            if type(g) == str
            else bonf_result.get(tuple(g.values), np.nan),
            axis=1,
        )
        data[fdr_name] = data[groupby].apply(
            lambda g: bonf_result.get(g, np.nan)
            if type(g) == str
            else fdr_result.get(tuple(g.values), np.nan),
            axis=1,
        )
        # Sort
        data.sort_values(by=[fdr_name, bonf_name], inplace=True)

        # Restore index
        data.set_index(index_cols, inplace=True)
