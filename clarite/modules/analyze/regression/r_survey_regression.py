from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from clarite.internal.utilities import requires, _get_dtypes

from .base import Regression


class RSurveyRegression(Regression):
    """
    Run regressions by calling R from Python
    When a SurveyDesignSpec is provided, the R *survey* library is used.
    Results should match those run with either GLMRegression or WeightedGLMRegression.

    Parameters
    ----------
    data: pd.DataFrame
    outcome_variable: str
      The variable to be used as the output (y) of the regression
    covariates: List[str]
      The variables to be used as covariates.  Any variables in the DataFrames not listed as covariates are regressed.
    min-n: int or None
      Minimum number of complete-case observations (no NA values for phenotype, covariates, variable, or weight)
      Defaults to 200
    survey_design_spec: SurveyDesignSpec or None
        A SurveyDesignSpec object is used to create SurveyDesign objects for each regression.

    Returns
    -------
    df: pd.DataFrame
        Results DataFrame with these columns:
        ['variable_type', 'N', 'beta', 'SE', 'var_pvalue', 'LRT_pvalue', 'diff_AIC', 'pvalue']
    """
    def __init__(self, data, outcome_variable, covariates, min_n=200, survey_design_spec=None):

        # base class init
        # This takes in minimal regression params (data, outcome_variable, covariates) and
        # initializes additional parameters (outcome dtype, regression variables, error, and warnings)
        super().__init__(data=data,
                         outcome_variable=outcome_variable,
                         covariates=covariates)

        # Custom init involving kwargs passed to this regression
        self.min_n = min_n
        self.survey_design_spec = survey_design_spec

        # Note this runs the entire regression in R, returning a DataFrame instead of a dict.
        # Therefore, store the dataframe in self.result instead of a dict in self.results
        self.result = None

        # Name the index "ID" (SurveyDesignSpec already does this)
        self.data.index = self.data.index.rename("ID")

        # Ensure the data output type is compatible
        if self.outcome_dtype == 'categorical':
            raise NotImplementedError("Categorical Phenotypes are not yet supported for this type of regression.")
        elif self.outcome_dtype == 'continuous':
            self.description += f"Continuous Outcome (family = Gaussian): '{self.outcome_variable}'"
            self.family = "gaussian"
        elif self.outcome_dtype == 'binary':
            # Use the order according to the categorical
            counts = self.data[self.outcome_variable].value_counts().to_dict()
            categories = self.data[self.outcome_variable].cat.categories
            codes, categories = zip(*enumerate(categories))
            self.data[self.outcome_variable].replace(categories, codes, inplace=True)
            self.description += f"Binary Outcome (family = Binomial): '{self.outcome_variable}'\n" \
                                f"\t{counts[categories[0]]:,} occurrences of '{categories[0]}' coded as 0\n" \
                                f"\t{counts[categories[1]]:,} occurrences of '{categories[1]}' coded as 1"
            self.family = "binomial"
        else:
            raise ValueError("The outcome variable's type could not be determined.  Please report this error.")

        # Log missing outcome values
        na_outcome_count = self.data[self.outcome_variable].isna().sum()
        self.description += f"\nUsing {len(self.data) - na_outcome_count:,} of {len(self.data):,} observations"
        if na_outcome_count > 0:
            self.description += f"\n\t{na_outcome_count:,} are missing a value for the outcome variable"

        # Finish updating description
        self.description += f"\nRegressing {len(self.results):,} variables"
        for k, v in self.regression_variables.items():
            self.description += f"\n\t{len(v):,} {k} variables"

    def get_results(self) -> Tuple[pd.DataFrame, Dict[str, List[str]], Dict[str, str]]:
        """
        Merge results into a dataFrame
        """
        if not self.run_complete:
            raise ValueError("No results: either the 'run' method was not called, or there was a problem running")

        return self.result

    @requires('rpy2')
    def run(self):
        """
        Run the regression using R
        """
        # Source R script to define the function
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from .r_code.r_utilities import ewasresult2py, df_pandas2r
        r_code_folder = (Path(__file__).parent / 'r_code')
        filename = str(r_code_folder / "ewas_r.R")
        ro.r.source(filename)

        # Print warnings as they occur
        ro.r("options(warn=1)")

        # Lists of regression variables (NULL if empty)
        bin_vars = ro.StrVector(self.regression_variables['binary'])
        cat_vars = ro.StrVector(self.regression_variables['categorical'])
        cont_vars = ro.StrVector(self.regression_variables['continuous'])
        if len(bin_vars) == 0:
            bin_vars = ro.NULL
        if len(cat_vars) == 0:
            cat_vars = ro.NULL
        if len(cont_vars) == 0:
            cont_vars = ro.NULL

        # Lists of covariates (NULL if empty)
        dtypes = _get_dtypes(self.data)
        bin_covars = ro.StrVector([v for v in self.covariates if (dtypes.loc[v] == 'binary')])
        cat_covars = ro.StrVector([v for v in self.covariates if (dtypes.loc[v] == 'categorical')])
        cont_covars = ro.StrVector([v for v in self.covariates if dtypes.loc[v] == 'continuous'])
        if len(bin_covars) == 0:
            bin_covars = ro.NULL
        if len(cat_covars) == 0:
            cat_covars = ro.NULL
        if len(cont_covars) == 0:
            cont_covars = ro.NULL

        # Allow nonvarying covariates by default to match python ewas (warn instead of error)
        allowed_nonvarying = ro.StrVector(self.covariates)

        # Run with or without survey design info
        if self.survey_design_spec is None:
            # Reset the index on data so that the first column is "ID" (note 'data' becomes a local variable)
            data = self.data.reset_index(drop=False)
            data = data[["ID", ] + [c for c in data.columns if c != "ID"]]

            with ro.conversion.localconverter(ro.default_converter + pandas2ri.converter):
                data_r = df_pandas2r(data)
                result = ro.r.ewas(d=data_r, bin_vars=bin_vars, cat_vars=cat_vars, cont_vars=cont_vars,
                                   y=self.outcome_variable,
                                   bin_covars=bin_covars, cat_covars=cat_covars, cont_covars=cont_covars,
                                   regression_family=self.family,
                                   allowed_nonvarying=allowed_nonvarying,
                                   min_n=self.min_n)
        else:
            # Merge weights into data and get weight name(s) (Note 'data' becomes a local variable)
            if self.survey_design_spec.single_weight:
                weights = self.survey_design_spec.weight_name
                data = pd.merge(self.data, self.survey_design_spec.weight_values, left_index=True, right_index=True, how='left')
            elif self.survey_design_spec.multi_weight:
                weights = self.survey_design_spec.weight_names
                data = pd.merge(self.data, pd.DataFrame(self.survey_design_spec.weight_values),
                                left_index=True, right_index=True, how='left')
            else:
                raise ValueError("Weights must be provided")
            # Gather optional parts of survey parameters
            kwargs = dict()
            # Cluster IDs
            if self.survey_design_spec.has_cluster:
                kwargs['ids'] = f"{self.survey_design_spec.cluster_name}"
                data[self.survey_design_spec.cluster_name] = self.survey_design_spec.cluster_values
            else:
                kwargs['ids'] = ro.NULL
            # Strata
            if self.survey_design_spec.has_strata:
                kwargs['strata'] = f"{self.survey_design_spec.strata_name}"
                data[self.survey_design_spec.strata_name] = self.survey_design_spec.strata_values
            # fpc
            if self.survey_design_spec.has_fpc:
                kwargs['fpc'] = f"{self.survey_design_spec.fpc_name}"
                data[self.survey_design_spec.fpc_name] = self.survey_design_spec.fpc_values_original

            # Single cluster setting
            ro.r(f'options("survey.lonely.psu"="{self.survey_design_spec.single_cluster}")')

            # Reset the index on data so that the first column is "ID"
            data = data.reset_index(drop=False)
            data = data[["ID", ] + [c for c in data.columns if c != "ID"]]

            with ro.conversion.localconverter(ro.default_converter + pandas2ri.converter):
                data_r = df_pandas2r(data)

                if self.survey_design_spec.multi_weight:
                    # Must convert python dict of var:weight name to a named list in R
                    weights = ro.ListVector(weights)

                result = ro.r.ewas(d=data_r, bin_vars=bin_vars, cat_vars=cat_vars, cont_vars=cont_vars,
                                   y=self.outcome_variable,
                                   bin_covars=bin_covars, cat_covars=cat_covars, cont_covars=cont_covars,
                                   regression_family=self.family,
                                   allowed_nonvarying=allowed_nonvarying,
                                   min_n=self.min_n,
                                   weights=weights,
                                   subset=self.survey_design_spec.subset_array,
                                   **kwargs)

        result = ewasresult2py(result)

        # Ensure correct dtypes (float may be objects if the are all NaN)
        float_cols = ['Beta', 'SE', 'Variable_pvalue', 'LRT_pvalue', 'Diff_AIC', 'pvalue']
        result[float_cols] = result[float_cols].astype('float64')

        self.result = result.reset_index(drop=False)
        self.run_complete = True
