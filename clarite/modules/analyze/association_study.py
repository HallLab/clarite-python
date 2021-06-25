from typing import Optional, Union, Type, List

import click
import pandas as pd

from clarite.modules.analyze import regression
from clarite.modules.analyze.regression import builtin_regression_kinds


def association_study(
    data: pd.DataFrame,
    outcomes: Union[str, List[str]],
    regression_variables: Optional[Union[str, List[str]]] = None,
    covariates: Optional[Union[str, List[str]]] = None,
    regression_kind: Union[str, Type[regression.Regression]] = "glm",
    encoding: Optional[str] = None,
    weighted_encoding_info: Optional[pd.DataFrame] = None,
    **kwargs,
):
    """
    Run an association study (EWAS, PhEWAS, GWAS, GxEWAS, etc)

    Individual regression classes selected with `regression_kind` may work slightly differently.
    Results are sorted in order of increasing `pvalue`

    Parameters
    ----------
    data: pd.DataFrame
        Contains all outcomes, regression_variables, and covariates
    outcomes: str or List[str]
        The exogenous variable (str) or variables (List) to be used as the output of each regression.
    regression_variables: str, List[str], or None
        The endogenous variable (str) or variables (List) to be used invididually as inputs into regression.
        If None, use all variables in `data` that aren't an outcome or a covariate
    covariates: str, List[str], or None (default)
        The variable (str) or variables (List) to be used as covariates in each regression.
    regression_kind: str or subclass of Regression, 'glm' by default
        This can be 'glm', 'weighted_glm', or 'r_survey' for built-in Regression types,
        or a custom subclass of Regression
    encoding: Optional[str], default None
        Encoding method to use for any genotype data.  One of {'additive', 'dominant', 'recessive', 'codominant', or 'weighted'}
    weighted_encoding_info: Optional pd.DataFrame, default None
        If weighted encoding is used, this must be provided.  See Pandas-Genomics documentation on weighted encodings.
    kwargs: Keyword arguments specific to the Regression being used

    Returns
    -------
    df: pd.DataFrame
        Association Study results DataFrame with at least these columns: ['N', 'pvalue', 'error', 'warnings'].
        Indexed by the outcome variable and the variable being assessed in each regression
    """
    # Copy data to avoid modifying the original, in case it is changed
    data = data.copy(deep=True)

    # Encode any genotype data
    if encoding is not None:
        if encoding == "additive":
            data = data.genomics.encode_additive()
        elif encoding == "dominant":
            data = data.genomics.encode_dominant()
        elif encoding == "recessive":
            data = data.genomics.encode_recessive()
        elif encoding == "codominant":
            data = data.genomics.encode_codominant()
        elif encoding == "weighted":
            if weighted_encoding_info is None:
                raise ValueError(
                    "'weighted_encoding_info' must be provided when using weighted encoding"
                )
            else:
                data = data.genomics.encode_weighted(weighted_encoding_info)
        else:
            raise ValueError(f"Unknown 'encoding': {encoding}")

    # Ensure outcome, covariates, and regression variables are strings
    if isinstance(outcomes, str):
        outcome = [
            outcomes,
        ]
    if isinstance(covariates, str):
        covariates = [
            outcome,
        ]
    if isinstance(regression_variables, str):
        regression_variables = [
            regression_variables,
        ]
    elif regression_variables is None:
        regression_variables = list(set(data.columns) - set(outcome) - set(covariates))

    # Parse regression kind
    if isinstance(regression_kind, str):
        regression_cls = builtin_regression_kinds.get(regression_kind, None)
        if regression_cls is None:
            raise ValueError(
                f"Unknown regression kind '{regression_kind}, known values are {','.join(builtin_regression_kinds.keys())}"
            )
    elif regression_kind in regression_kind.mro():
        regression_cls = regression_kind
    else:
        raise ValueError(
            f"Incorrect regression kind type ({type(regression_kind)}).  "
            f"A valid string or a subclass of Regression is required."
        )

    # Run each regression
    results = []
    for outcome in outcomes:
        regression = regression_cls(
            data=data,
            outcome_variable=outcome,
            regression_variables=regression_variables,
            covariates=covariates,
            **kwargs,
        )
        print(regression)

        # Run and get results
        regression.run()
        result = regression.get_results()

        # Process Results
        click.echo(f"Completed Association Study for {outcome}\n", color="green")
        results.append(result)

    if len(outcomes) == 1:
        result = results[0]
    else:
        result = pd.concat(results)

    # Sort across multiple outcomes
    if result.index.names == ["Variable", "Outcome", "Category"]:
        result = result.sort_values(["pvalue", "Beta_pvalue"])
    elif result.index.names == ["Variable", "Outcome"]:
        result = result.sort_values(["pvalue"])

    click.echo("Completed association study", color="green")
    return result
