from typing import List, Optional, Tuple

import pandas as pd
import click

from .regression import InteractionRegression


def interactions(outcome_variable: str,
                 covariates: List[str],
                 data: pd.DataFrame,
                 min_n: int = 200,
                 interactions: Optional[List[Tuple[str, str]]] = None,
                 report_betas: bool = False):
    """
    Run an Environment-Wide Association Study

    All variables in `data` other than the outcome and covariates are potential interaction variables.
    All pairwise interactions are tested unless specific
    Results are sorted in order of increasing `pvalue`

    Parameters
    ----------
    data: pd.DataFrame
        The data to be analyzed, including the phenotype, covariates, and any variables to be regressed.
    outcome_variable: string
        The variable to be used as the output (y) of the regression
    covariates: list (strings),
        The variables to be used as covariates.  Any variables in the DataFrames not listed as covariates are regressed.
    min_n: int or None
        Minimum number of complete-case observations (no NA values for phenotype, covariates, or variable)
        Defaults to 200
    interactions: list(tuple(strings)) or None
        A list of tuples of variable names to be tested as interactions.  If not specified, all pairwise interactions
        will be tested for any variables in the data that are not the outcome or covariates.
    report_betas: boolean
        False by default.
          If True, the results will contain one row for each interaction term and will include the beta value
          for that term.  The number of terms increases with the number of categories in each interacting term.


    Returns
    -------
    df: pd.DataFrame
        DataFrame with at least these columns: ['N', 'pvalue', 'error', 'warnings']
        indexed by the interaction and the outcome

    Examples
    --------
    >>> ewas_discovery = clarite.analyze.test_interactions("logBMI", covariates, nhanes_discovery)
    Running EWAS on a continuous variable
    """
    # Copy data to avoid modifying the original, in case it is changed
    data = data.copy(deep=True)

    # Initialize the regression and print details
    regression = InteractionRegression(data=data,
                                       outcome_variable=outcome_variable,
                                       covariates=covariates,
                                       min_n=min_n,
                                       interactions=interactions,
                                       report_betas=report_betas)
    print(regression)

    # Run and get results
    regression.run()
    result = regression.get_results()

    # Process Results
    result['Outcome'] = outcome_variable
    result = result.sort_values('LRT_pvalue').set_index(['Interaction', 'Outcome'])  # Sort and set index
    column_order = ['Test_Number', 'Converged', 'N', 'Beta', 'SE', 'Beta_pvalue', 'LRT_pvalue']
    result = result[column_order]  # Sort columns
    click.echo("Completed Interaction Analysis\n")
    return result
