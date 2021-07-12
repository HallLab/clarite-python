from typing import List, Optional, Type, Union, Any

import click

from clarite.modules.analyze import regression
from clarite.modules.analyze.regression import builtin_regression_kinds


def ewas(
    outcome: str,
    covariates: List[str],
    data: Any,
    regression_kind: Optional[Union[str, Type[regression.Regression]]] = None,
    **kwargs,
):
    """
    Run an Environment-Wide Association Study

    All variables in `data` other than the outcome (outcome) and covariates are tested individually.
    Individual regression classes selected with `regression_kind` may work slightly differently.
    Results are sorted in order of increasing `pvalue`

    Parameters
    ----------
    outcome: string
        The variable to be used as the output of the regressions
    covariates: list (strings),
        The variables to be used as covariates.  Any variables in the DataFrames not listed as covariates are regressed.
    data: Any, usually pd.DataFrame
        The data to be analyzed, including the outcome, covariates, and any variables to be regressed.
    regression_kind: str or subclass of Regression
        This can be 'glm', 'weighted_glm', or 'r_survey' for built-in Regression types,
        or a custom subclass of Regression
        None by default to maintain existing api (`glm` unless SurveyDesignSpec exists, in which case `weighted_glm`)
    kwargs: Keyword arguments specific to the Regression being used

    Returns
    -------
    df: pd.DataFrame
        EWAS results DataFrame with at least these columns: ['N', 'pvalue', 'error', 'warnings']
        indexed by the outcome and the variable being assessed in each row

    Examples
    --------
    >>> ewas_discovery = clarite.analyze.ewas("logBMI", covariates, nhanes_discovery)
    Running on a continuous variable
    """
    raise DeprecationWarning(
        "This function will be depreciated in favor of clarite.analyze.association_study"
    )
    # Copy data to avoid modifying the original, in case it is changed
    data = data.copy(deep=True)

    # Set up regression object
    # Emulate existing API by figuring out which method automatically
    # glm if not specified, unless survey_design_spec is passed and isn't None
    if regression_kind is None:
        if "survey_design_spec" in kwargs:
            if kwargs["survey_design_spec"] is None:
                regression_kind = "glm"
                del kwargs["survey_design_spec"]
            else:
                regression_kind = "weighted_glm"
        else:
            regression_kind = "glm"

    if type(regression_kind) == str:
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

    # regression variables are anything that isn't an outcome or covariate
    regression_variables = set(data.columns) - set(
        [
            outcome,
        ]
        + covariates
    )

    # Initialize the regression and print details
    print(kwargs)
    regression = regression_cls(
        data=data,
        outcome_variable=outcome,
        covariates=covariates,
        regression_variables=regression_variables,
        **kwargs,
    )
    print(regression)

    # Run and get results
    regression.run()
    result = regression.get_results()

    # Process Results
    click.echo("Completed EWAS\n")
    return result
