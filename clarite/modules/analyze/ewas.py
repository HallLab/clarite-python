from typing import List, Optional, Type, Union, Any

import click

from clarite.modules.analyze import regression

builtin_regression_kinds = {
    'glm': regression.GLMRegression,
    'weighted_glm': regression.WeightedGLMRegression,
    'r_survey': regression.RSurveyRegression
}


def ewas(phenotype: str,
         covariates: List[str],
         data: Any,
         regression_kind: Optional[Union[str, Type[regression.Regression]]] = None,
         **kwargs):
    """
    Run an Environment-Wide Association Study

    All variables in `data` other than the phenotype (outcome) and covariates are tested individually.
    Individual regression classes selected with `regression_kind` may work slightly differently.
    Results are sorted in order of increasing `pvalue`

    Parameters
    ----------
    phenotype: string
        The variable to be used as the output of the regressions
    covariates: list (strings),
        The variables to be used as covariates.  Any variables in the DataFrames not listed as covariates are regressed.
    data: Any, usually pd.DataFrame
        The data to be analyzed, including the phenotype, covariates, and any variables to be regressed.
    regression_kind: str or subclass of Regression
        This can be 'glm', 'glm_weighted', or 'r_survey' for built-in Regression types,
        or a custom subclass of Regression
        None by default to maintain existing api (`glm` unless SurveyDesignSpec exists, in which case `weighted_glm`)
    kwargs: Keyword arguments specific to the Regression being used

    Returns
    -------
    df: pd.DataFrame
        EWAS results DataFrame with at least these columns: ['N', 'pvalue', 'error', 'warnings']
        indexed by the phenotype/outcome and the variable being assessed in each row

    Examples
    --------
    >>> ewas_discovery = clarite.analyze.ewas("logBMI", covariates, nhanes_discovery)
    Running EWAS on a continuous variable
    """
    # Copy data to avoid modifying the original, in case it is changed
    data = data.copy(deep=True)

    # Set up regression object
    # Emulate existing API by figuring out which method automatically
    if regression_kind is None:
        if 'survey_design_spec' in kwargs:
            regression_kind = 'weighted_glm'
        else:
            regression_kind = 'glm'

    if type(regression_kind) == str:
        regression_cls = builtin_regression_kinds.get(regression_kind, None)
        if regression_cls is None:
            raise ValueError(f"Unknown regression kind '{regression_kind}")
    elif regression_kind in regression_kind.mro():
        regression_cls = regression_kind
    else:
        raise ValueError(f"Incorrect regression kind type ({type(regression_kind)}).  "
                         f"A valid string or a subclass of Regression is required.")

    # Initialize the regression and print details
    print(kwargs)
    regression = regression_cls(data=data,
                                outcome_variable=phenotype,
                                covariates=covariates,
                                **kwargs)
    print(regression)

    # Run and get results
    regression.run()
    result = regression.get_results()

    # Process Results
    result['Phenotype'] = phenotype  # Add phenotype
    result = result.sort_values('pvalue').set_index(['Variable', 'Phenotype'])  # Sort and set index
    if 'Weight' not in result.columns:
        result['Weight'] = None
    column_order = ['Variable_type', 'Weight', 'Converged', 'N', 'Beta', 'SE',
                    'Variable_pvalue', 'LRT_pvalue', 'Diff_AIC', 'pvalue']
    result = result[column_order]  # Sort columns
    click.echo("Completed EWAS\n")
    return result
