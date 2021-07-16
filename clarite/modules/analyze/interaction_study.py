from typing import List, Optional, Tuple, Union

import pandas as pd
import click
from pandas_genomics import GenotypeDtype

from .regression import InteractionRegression


def interaction_study(
    data: pd.DataFrame,
    outcomes: Union[str, List[str]],
    interactions: Optional[Union[List[Tuple[str, str]], str]] = None,
    covariates: Optional[Union[str, List[str]]] = None,
    encoding: str = "additive",
    edge_encoding_info: Optional[pd.DataFrame] = None,
    report_betas: bool = False,
    min_n: int = 200,
    process_num: Optional[int] = None,
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
    outcomes: str or List[str]
        The exogenous variable (str) or variables (List) to be used as the output of each regression.
    interactions: list(tuple(strings)), str, or None
        Valid variables are those in the data that are not an outcome variable or a covariate.
        None: Test all pairwise interactions between valid variables
        String: Test all interactions of this valid variable with other valid variables
        List of tuples: Test specific interactions of valid variables
    covariates: str, List[str], or None (default)
        The variable (str) or variables (List) to be used as covariates in each regression.
    encoding: str, default "additive""
        Encoding method to use for any genotype data.  One of {'additive', 'dominant', 'recessive', 'codominant', or 'edge'}
    edge_encoding_info: Optional pd.DataFrame, default None
        If edge encoding is used, this must be provided.  See Pandas-Genomics documentation on edge encoding.
    report_betas: boolean
        False by default.
          If True, the results will contain one row for each interaction term and will include the beta value,
          standard error (SE), and beta pvalue for that specific interaction. The number of terms increases with
          the number of categories in each interacting variable.
    min_n: int or None
        Minimum number of complete-case observations (no NA values for outcome, covariates, or variable)
        Defaults to 200
    process_num: Optional[int]
        Number of processes to use when running the analysis, default is None (use the number of cores)

    Returns
    -------
    df: pd.DataFrame
        DataFrame with these columns: ['Test_Number', 'Converged', 'N', 'Beta', 'SE', 'Beta_pvalue', 'LRT_pvalue']
        indexed by the interaction terms ("Term1", "Term2") and the outcome variable ("Outcome")
    """
    # Copy data to avoid modifying the original, in case it is changed
    data = data.copy(deep=True)

    # Encode any genotype data
    has_genotypes = False
    for dt in data.dtypes:
        if GenotypeDtype.is_dtype(dt):
            has_genotypes = True
            break
    if has_genotypes:
        if encoding == "additive":
            data = data.genomics.encode_additive()
        elif encoding == "dominant":
            data = data.genomics.encode_dominant()
        elif encoding == "recessive":
            data = data.genomics.encode_recessive()
        elif encoding == "codominant":
            data = data.genomics.encode_codominant()
        elif encoding == "edge":
            if edge_encoding_info is None:
                raise ValueError(
                    "'edge_encoding_info' must be provided when using edge encoding"
                )
            else:
                data = data.genomics.encode_edge(edge_encoding_info)
        else:
            raise ValueError(f"Genotypes provided with unknown 'encoding': {encoding}")

    # Ensure outcome, covariates, and regression variables are lists
    if isinstance(outcomes, str):
        outcomes = [
            outcomes,
        ]
    if isinstance(covariates, str):
        covariates = [
            covariates,
        ]
    elif covariates is None:
        covariates = []

    # Run each regression
    results = []
    for outcome in outcomes:
        regression = InteractionRegression(
            data=data,
            outcome_variable=outcome,
            covariates=covariates,
            min_n=min_n,
            interactions=interactions,
            report_betas=report_betas,
            process_num=process_num,
        )
        print(regression)

        # Run and get results
        regression.run()
        result = regression.get_results()

        # Process Results
        click.echo(f"Completed Interaction Study for {outcome}\n", color="green")
        results.append(result)

    if len(outcomes) == 1:
        result = results[0]
    else:
        result = pd.concat(results)

    # Sort across multiple outcomes
    result = result.sort_values(["LRT_pvalue", "Beta_pvalue"])

    click.echo("Completed association study", color="green")
    return result
