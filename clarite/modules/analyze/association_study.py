from typing import Optional, Union, Type, List

import click
import pandas as pd

from clarite.modules.analyze import regression
from clarite.modules.analyze.regression import (
    builtin_regression_kinds,
    WeightedGLMRegression,
    GLMRegression,
)


def association_study(
    data: pd.DataFrame,
    outcomes: Union[str, List[str]],
    regression_variables: Optional[Union[str, List[str]]] = None,
    covariates: Optional[Union[str, List[str]]] = None,
    regression_kind: Optional[Union[str, Type[regression.Regression]]] = None,
    encoding: str = "additive",
    edge_encoding_info: Optional[pd.DataFrame] = None,
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
    regression_kind: None, str or subclass of Regression
        This can be 'glm', 'weighted_glm', or 'r_survey' for built-in Regression types,
        or a custom subclass of Regression.  If None, it is set to 'glm' if a survey design is not specified
        and 'weighted_glm' if it is.
    kwargs: Keyword arguments specific to the Regression being used

    Returns
    -------
    df: pd.DataFrame
        Association Study results DataFrame with at least these columns: ['N', 'pvalue', 'error', 'warnings'].
        Indexed by the outcome variable and the variable being assessed in each regression
    """
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
    if isinstance(regression_variables, str):
        regression_variables = [
            regression_variables,
        ]
    elif regression_variables is None:
        regression_variables = list(set(data.columns) - set(outcomes) - set(covariates))

    # Delete the survey_design_spec kwarg if it is None
    # This would be fine, but kwarg parsing for different clases means possibly passing it to an init that isn't expecting it
    if "survey_design_spec" in kwargs:
        if kwargs["survey_design_spec"] is None:
            del kwargs["survey_design_spec"]

    # Parse regression kind
    if regression_kind is None:
        # Match the original api, which is glm or weighted_glm based on whether a design is passes
        if "survey_design_spec" in kwargs:
            regression_cls = WeightedGLMRegression
        else:
            regression_cls = GLMRegression
    elif isinstance(regression_kind, str):
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
