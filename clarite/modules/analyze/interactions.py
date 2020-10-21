from typing import List, Optional, Tuple, Union

import pandas as pd
import click

from .regression import InteractionRegression


def interaction_test(
    outcome_variable: str,
    covariates: List[str],
    data: pd.DataFrame,
    min_n: int = 200,
    interactions: Optional[Union[List[Tuple[str, str]], str]] = None,
    report_betas: bool = False,
):
    """Perform LRT tests comparing a model with interaction terms to one without.

    An intercept, covariates, and main effects of the variables used in the interactiona are included in both the full
    and restricted models.
    All variables in `data` other than the outcome and covariates are potential interaction variables.
    All pairwise interactions are tested unless specific
    Results are sorted in order of increasing `pvalue`

    Parameters
    ----------
    data: pd.DataFrame
        The data to be analyzed, including the outcome, covariates, and any variables to be regressed.
    outcome_variable: string
        The variable to be used as the output (y) of the regression
    covariates: list (strings),
        The variables to be used as covariates.  Any variables in the DataFrames not listed as covariates are regressed.
    min_n: int or None
        Minimum number of complete-case observations (no NA values for outcome, covariates, or variable)
        Defaults to 200
    interactions: list(tuple(strings)), str, or None
        Valid variables are those in the data that are not the outcome variable or a covariate.
        None: Test all pairwise interactions between valid variables
        String: Test all interactions of this valid variable with other valid variables
        List of tuples: Test specific interactions of valid variables
    report_betas: boolean
        False by default.
          If True, the results will contain one row for each interaction term and will include the beta value,
          standard error (SE), and beta pvalue for that specific interaction. The number of terms increases with
          the number of categories in each interacting variable.


    Returns
    -------
    df: pd.DataFrame
        DataFrame with these columns: ['Test_Number', 'Converged', 'N', 'Beta', 'SE', 'Beta_pvalue', 'LRT_pvalue']
        indexed by the interaction terms ("Term1", "Term2") and the outcome variable ("Outcome")

    Examples
    --------
    >>> ewas_discovery = clarite.analyze.interaction_test("logBMI", covariates, nhanes_discovery)
    """
    # Copy data to avoid modifying the original, in case it is changed
    data = data.copy(deep=True)

    # Initialize the regression and print details
    regression = InteractionRegression(
        data=data,
        outcome_variable=outcome_variable,
        covariates=covariates,
        min_n=min_n,
        interactions=interactions,
        report_betas=report_betas,
    )
    print(regression)

    # Run and get results
    regression.run()
    result = regression.get_results()

    click.echo("Completed Interaction Analysis\n")
    return result
