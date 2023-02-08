from typing import Dict, Optional, Tuple, Union

import click
import numpy as np
import pandas as pd
from pandas.core.indexing import IndexingError


class SurveyDesignSpec:
    """
    Holds parameters for building a statsmodels SurveyDesign object

    Parameters
    ----------
    survey_df: pd.DataFrame
        A DataFrame containing Cluster, Strata, and/or weights data.
        This should include all observations in the data analyzed using it (matching via index value)
    strata: string or None
        The name of the strata variable in the survey_df
    cluster: string or None
        The name of the cluster variable in the survey_df
    nest: bool, default False
        Whether or not the clusters are nested in the strata (The same cluster IDs are repeated in different strata)
    weights: string or dictionary(string:string)
        The name of the weights variable in the survey_df, or a dictionary mapping variable names to weight names
    fpc: string or None
        The name of the variable in the survey_df that contains the finite population correction information.
        This reduces variance when a substantial portion of the population is sampled.
        May be specified as the total population size, or the fraction of the population that was sampled.
    single_cluster: {'fail', 'adjust', 'average', 'certainty'}
        Setting controlling variance calculation in single-cluster ('lonely psu') strata
        'fail': default, throw an error
        'adjust': use the average of all observations (more conservative)
        'average': use the average value of other strata
        'certainty': that strata doesn't contribute to the variance (0 variance)
    drop_unweighted: bool, default False
        If True, drop observations that are missing a weight value.  This may not be statistically sound.
        Otherwise the result for variables with missing weights (when the variable is not missing) is NULL.

    Attributes
    ----------

    Examples
    --------
    >>> import clarite
    >>> clarite.analyze.SurveyDesignSpec(survey_df=survey_design_replication,
                                         strata="SDMVSTRA",
                                         cluster="SDMVPSU",
                                         nest=True,
                                         weights=weights_replication,
                                         fpc=None,
                                         single_cluster='fail')
    """

    def __init__(
        self,
        survey_df: pd.DataFrame,
        strata: Optional[str] = None,
        cluster: Optional[str] = None,
        nest: bool = False,
        weights: Union[str, Dict[str, str]] = None,
        fpc: Optional[str] = None,
        single_cluster: Optional[str] = "fail",
        drop_unweighted: bool = False,
    ):

        # Validate index
        if isinstance(survey_df.index, pd.MultiIndex):
            raise ValueError("survey_df: DataFrame must not have a multiindex")
        survey_df.index.name = "ID"
        self.index = survey_df.index

        # At least one must be defined
        if all([x is None for x in (strata, cluster, weights)]):
            raise ValueError(
                """At least one of strata, cluster, or weights must be provided"""
            )

        # Store parameters
        self.drop_unweighted = drop_unweighted
        self.single_cluster = single_cluster
        # Warn if drop_unweighted is set to True
        if self.drop_unweighted:
            click.echo(
                click.style(
                    "WARNING: Dropping observations with missing weights. "
                    "This may not be statistically sound, and the cause of missing weights "
                    "should be determined.",
                    fg="red",
                )
            )
        # Validate single_cluster parameter
        if single_cluster not in {"fail", "adjust", "average", "certainty"}:
            raise ValueError(
                "'single_cluster' must be one of 'fail', 'adjust', 'average', or 'certainty'."
            )

        # Defaults
        # Strata
        self.has_strata: bool = False
        self.strata_name: Optional[str] = None
        self.strata_values: Optional[pd.Series] = None
        self.n_strat: Optional[int] = None
        # Cluster
        self.has_cluster: bool = False
        self.cluster_name: Optional[str] = None
        self.cluster_values: Optional[pd.Series] = None
        self.n_clust: Optional[int] = None
        self.nested_clusters: bool = False
        # Weight
        self.single_weight: bool = False  # If True, weight_values is a Series
        self.weight_name: Optional[str] = None
        self.multi_weight: bool = (
            False  # If True, weight_values is a dict of weight name : Series
        )
        self.weight_names: Optional[
            Dict[str, str]
        ] = None  # Dict of regression variable name : weight name
        self.weight_values: Optional[Union[pd.Series, Dict[str, pd.Series]]] = None
        # FPC
        self.has_fpc: bool = False
        self.fpc_name: Optional[str] = None
        self.fpc_original: Optional[pd.Series] = None
        self.fpc_values: Optional[pd.Series] = None

        # Process inputs
        self.process_strata(strata, survey_df)
        self.process_clusters(cluster, survey_df, nest)
        self.process_weights(weights, survey_df)
        self.process_fpc(fpc, survey_df)

        # Map relationships between inputs
        combined = pd.concat(
            [self.strata_values, self.cluster_values, self.fpc_values], axis=1
        )
        # The number of clusters per stratum
        self.clust_per_strat = combined.groupby("strat")["clust"].nunique()
        # The stratum for each cluster
        self.strat_for_clust = (
            combined.groupby("clust")["strat"].unique().apply(lambda l: l[0])
        )

        # Initialize the subset information (a boolean array of True, indicating every row is kept)
        self.subset_array = pd.Series(True, index=survey_df.index, name="subset")
        self.subset_count = 0

        # Raise an error if single clusters were found but shouldn't be allowed
        if (
            self.has_strata
            and self.has_cluster
            and self.single_cluster not in {"average", "certainty", "adjust"}
        ):
            if self.clust_per_strat.min() < 2:
                single_clusters = [
                    str(c)
                    for c in self.clust_per_strat[
                        self.clust_per_strat == 1
                    ].index.values
                ]
                raise ValueError(
                    f"One or more strata have single clusters: {', '.join(single_clusters)}. "
                    f"Adjust the 'single_cluster' SurveyDesignSpec parameter "
                    f"or reassign the singular cluster to avoid this error."
                )

    def process_strata(self, strata, survey_df):
        """
        Load Strata or generate default values
        """
        if strata is None:
            self.strata_values = pd.Series(
                np.ones(len(self.index)), index=self.index, name="strat"
            )
        else:
            self.strata_name = strata
            self.has_strata = True
            if strata not in survey_df:
                raise KeyError(
                    f"strata key ('{strata}') was not found in the survey_df"
                )
            elif survey_df[strata].isna().any():
                raise ValueError(
                    f"{survey_df[strata].isna().sum():,} of {len(survey_df):,} strata values were missing"
                )
            else:
                self.strata_values = survey_df[strata].rename("strat")
            self.n_strat = len(self.strata_values.unique())

        # Make categorical
        self.strata_values = self.strata_values.astype("category")

    def process_clusters(self, cluster, survey_df, nest):
        """
        Load clusters or generate default values
        """
        if cluster is None:
            self.cluster_values = pd.Series(
                np.arange(len(self.index)), index=self.index, name="clust"
            )
        else:
            self.cluster_name = cluster
            self.has_cluster = True
            if cluster not in survey_df:
                raise KeyError(
                    f"cluster key ('{cluster}') was not found in the survey_df"
                )
            elif survey_df[cluster].isna().any():
                raise ValueError(
                    f"{survey_df[cluster].isna().sum():,} of {len(survey_df):,} "
                    f"cluster values were missing"
                )
            else:
                self.cluster_values = survey_df[cluster].rename("clust")
                self.n_clust = len(self.cluster_values.unique())

        # If 'nest', recode the PSUs to be sure that the same PSU ID in different strata are treated as distinct PSUs.
        if nest and self.has_strata and self.has_cluster:
            self.cluster_values = (
                self.strata_values.astype(str) + "-" + self.cluster_values.astype(str)
            )
            self.cluster_values = self.cluster_values.rename("clust")
            self.nested_clusters = True

        # Make categorical
        self.cluster_values = self.cluster_values.astype("category")

    def process_weights(self, weights, survey_df):
        if weights is None:
            self.weight_values = pd.Series(
                np.ones(len(self.index)), index=self.index, name="weights"
            )
        elif type(weights) == dict:
            # self.weight_values will be a dictionary of weight_name: Series of weight values
            self.multi_weight = True
            self.weight_names = weights
            self.weight_values = dict()  # dict of weight name : weight values
            for var_name, weight_name in weights.items():
                if weight_name not in survey_df:
                    # Raise an error if the weight wasn't found in the survey dataframe
                    raise KeyError(
                        f"weights key for '{var_name}' ('{weight_name}') was not found in the survey_df"
                    )
                elif weight_name not in self.weight_values:
                    # If it hasn't already been processed (for another variable)
                    # Replace zero/negative weights with a small number to avoid divide by zero
                    zero_weights = survey_df[weight_name] <= 0
                    survey_df.loc[zero_weights, weight_name] = 1e-99
                    self.weight_values[weight_name] = survey_df[weight_name]
        elif type(weights) == str:
            # self.weight_values will be a Series of weight values
            self.single_weight = True
            self.weight_name = weights
            if self.weight_name not in survey_df:
                raise KeyError(
                    f"the weight ('{self.weight_name}') was not found in the survey_df"
                )
            else:
                # Replace zero weights with a small number to avoid divide by zero
                zero_weights = survey_df[self.weight_name] <= 0
                survey_df.loc[zero_weights, self.weight_name] = 1e-99
                self.weight_values = survey_df[self.weight_name]
        else:
            raise ValueError(
                "'weight' must be None, a weight name string, or a dictionary"
                " mapping variable name strings to weight name strings"
            )

    def process_fpc(self, fpc, survey_df):
        """
        FPC is passed in as
        """
        if fpc is None:
            self.fpc_values = pd.Series(
                np.zeros(len(self.index)), index=self.index, name="fpc"
            )
        else:
            # TODO: Should fpc scaling occuring after subsetting?
            self.fpc_name = fpc
            self.has_fpc = True
            if fpc not in survey_df:
                raise KeyError(f"fpc key ('{fpc}') was not found in the survey_df")
            elif survey_df[fpc].isna().any():
                raise ValueError(
                    f"{survey_df[fpc].isna().sum():,} of {len(survey_df):,} fpc values were missing"
                )
            else:
                self.fpc_values_original = survey_df[fpc].rename(
                    "fpc"
                )  # Need unmodified version for R code
                self.fpc_values = survey_df[fpc].rename("fpc")
                # Validate
                if not all(self.fpc_values <= 1):
                    # Assume these are actual population size, and convert to a fraction
                    if self.has_strata:
                        # Divide the sampled strata size by the fpc
                        combined = pd.merge(
                            self.strata_values,
                            self.fpc_values,
                            left_index=True,
                            right_index=True,
                        )
                        sampled_strata_size = combined.groupby("strat")[
                            "fpc"
                        ].transform("size")
                        self.fpc_values = sampled_strata_size / self.fpc_values
                    elif self.has_cluster and not self.has_strata:
                        # Clustered sampling: Divide sampled clusters by the fpc
                        sampled_cluster_size = len(self.cluster_values.unique())
                        self.fpc_values = sampled_cluster_size / self.fpc_values
                    try:
                        assert all((self.fpc_values >= 0) & (self.fpc_values <= 1))
                    except AssertionError:
                        raise ValueError("Error processing FPC- invalid values")
        # Reindex to list fpc for each observation
        combined = pd.concat([self.cluster_values, self.fpc_values], axis=1)
        self.fpc_values = (
            combined.groupby("clust")["fpc"].unique().apply(lambda l: l[0])
        )

    def __str__(self):
        """String version of the survey design specification, used in logging"""
        result = (
            f"Survey Design\n\t{len(self.index):,} rows in the survey design data\n"
        )
        # Strata
        if self.has_strata:
            result += f"\tStrata: {len(self.strata_values.unique())} unique values of {self.strata_name}\n"
        else:
            result += "\tStrata: None\n"
        # Clusters
        if self.has_cluster:
            result += f"\tCluster: {len(self.cluster_values.unique())} unique values of {self.cluster_name}"
            if self.nested_clusters:
                result += " (nested)\n"
            else:
                result += "\n"
        else:
            result += "\tCluster: None\n"
        # FPC
        if self.has_fpc:
            result += f"\tFPC: {self.fpc_name}\n"
        else:
            result += "\tFPC: None\n"
        # Weights
        if self.single_weight:
            result += (
                f"\tWeight: {self.weight_name}\n"
                f"\tDrop Unweighted: {self.drop_unweighted}\n"
            )
        elif self.multi_weight:
            result += (
                f"\tMultiple Weights: {len(set(self.weight_names.values())):,} "
                f"unique weights associated with {len(set(self.weight_names.keys())):,} variables\n"
                f"\tDrop Unweighted: {self.drop_unweighted}\n"
            )
        else:
            result += "\tWeights: None\n"
        # single cluster
        result += f"\tSingle Cluster ('Lonely PSU') Option: {self.single_cluster}"

        result += (
            f"\n\tSubsets: {self.subset_count:,} applied"
            f"\n\t\tKeeping {self.subset_array.sum():,} of {len(self.subset_array):,} observations"
        )

        return result

    def get_weights(
        self, regression_variable: str
    ) -> Tuple[bool, Optional[str], Optional[pd.Series]]:
        """
        Return weight information for a specific regression variable, subset to match the data
        """
        if self.single_weight:
            has_weights = True
            weight_name = self.weight_name
            weight_values = self.weight_values
        elif self.multi_weight:
            has_weights = True
            weight_name = self.weight_names.get(regression_variable, None)
            if weight_name is None:
                raise ValueError(
                    f"No weight found in the survey design for the '{regression_variable}' variable"
                )
            else:
                weight_values = self.weight_values[weight_name]
        else:
            return False, None, None

        # Normalize weights before subsetting
        weight_values = weight_values.div(weight_values.mean())

        # Subset
        weight_values = weight_values.loc[self.subset_array]

        return has_weights, weight_name, weight_values

    def check_missing_weights(
        self, data: pd.DataFrame, regression_variable: str
    ) -> Tuple[Optional[str], Optional[pd.Series], Optional[str]]:
        """
        Return:
            None, None, None if no weights are used
            weight_name, None, None if there are no missing weights
            weight_name, missing_weight_mask, warning if there are missing weights and 'drop_unweighted' is True
            Raise an error if there are missing weights and 'drop_unweighted' is False

            missing_weight_mask is True if the weight is missing
        """
        # Get weight values
        has_weight, weight_name, weight_values = self.get_weights(regression_variable)
        if not has_weight:
            return None, None, None  # No idx and no warning needed

        # Check if the survey design is missing weights when the variable value is not
        variable_na = data[regression_variable].isna()
        weight_na = weight_values.isna()
        missing_weight_mask = ~variable_na & weight_na

        if missing_weight_mask.sum() == 0:
            return weight_name, missing_weight_mask, None
        elif missing_weight_mask.sum() > 0 and self.drop_unweighted:
            # Warn, Drop observations with missing weights, and re-validate (for nonvarying covariates, for example)
            warning = (
                f"Dropping {missing_weight_mask.sum():,} non-missing observation(s) due to missing weights"
                f" (f{weight_name})"
            )
            weight_name += (
                f" ({missing_weight_mask.sum()} observations are missing weights)"
            )
            return weight_name, missing_weight_mask, warning
        elif missing_weight_mask.sum() > 0 and not self.drop_unweighted:
            # Get unique values
            values_with_missing_weight = data.loc[
                missing_weight_mask, regression_variable
            ]
            unique_missing = values_with_missing_weight.unique()
            unique_not_missing = data.loc[
                ~variable_na & ~weight_na, regression_variable
            ].unique()
            sometimes_missing = sorted(
                [str(v) for v in (set(unique_missing) & set(unique_not_missing))]
            )
            always_missing = sorted(
                [str(v) for v in (set(unique_missing) - set(unique_not_missing))]
            )

            # Build a detailed error string
            error = (
                f"{missing_weight_mask.sum():,} observations are missing weights ({weight_name})"
                f" when the variable is not missing."
            )

            # Add more information to the error and raise it, skipping analysis of this variable
            if len(sometimes_missing) == 0:
                pass
            elif len(sometimes_missing) == 1:
                error += f"\n\tOne value sometimes occurs in observations with missing weight: {sometimes_missing[0]}"
            elif len(sometimes_missing) <= 5:
                error += (
                    f"\n\t{len(sometimes_missing)} values sometimes occur in observations with missing weight:"
                    f" {', '.join(sometimes_missing)}"
                )
            elif len(sometimes_missing) > 5:
                error += (
                    f"\n\t{len(sometimes_missing)} values sometimes occur in observations with missing weight:"
                    f" {', '.join(sometimes_missing[:5])}, ..."
                )
            # Log always missing values
            if len(always_missing) == 0:
                pass
            elif len(always_missing) == 1:
                error += (
                    f"\n\tOne value is only found in observations with missing weights: {always_missing[0]}."
                    " Should it be encoded as NaN?"
                )
            elif len(always_missing) <= 5:
                error += (
                    f"\n\t{len(always_missing)} values are only found in observations with missing weights: "
                    f"{', '.join(always_missing)}. Should they be encoded as NaN?"
                )
            elif len(always_missing) > 5:
                error += (
                    f"\n\t{len(always_missing)} values are only found in observations with missing weights: "
                    f"{', '.join(always_missing[:5])}, ... Should they be encoded as NaN?"
                )
            raise ValueError(error)

    def validate(self, data: pd.DataFrame) -> Optional[str]:
        """
        Validate that the survey design matches the data

        Parameters
        ----------
        data: pd.DataFrame
            Data being used with the survey design

        Returns
        -------
        error: str or None
            Any error message
        """
        # Check to see if survey information or weights are included in the data
        msg = " Survey design variables should not be included in the data."
        if self.has_strata:
            if self.strata_name in data.columns:
                return (
                    f"Strata variable ({self.strata_name}) found in the passed data."
                    + msg
                )
        if self.has_cluster:
            if self.cluster_name in data.columns:
                return (
                    f"Cluster variable ({self.cluster_name}) found in the passed data."
                    + msg
                )
        if self.has_fpc:
            if self.fpc_name in data.columns:
                return f"FPC variable ({self.fpc_name}) found in the passed data." + msg
        if self.single_weight:
            if self.weight_name in data.columns:
                return (
                    f"Weight variable ({self.weight_name}) found in the passed data."
                    + msg
                )
        if self.multi_weight:
            matched = set(self.weight_names.values()) & set(data.columns)
            if len(matched) == 1:
                return (
                    f"Weight variable ({list(matched)[0]}) found in the passed data."
                    + msg
                )
            if len(matched) > 1:
                return (
                    f"{len(matched):,} Weight variables found in the passed data." + msg
                )
        # Validate that subsets apply to the data
        if self.subset_count > 0:
            try:
                data = data.loc[self.subset_array]
            except Exception as e:
                return f"Error applying the subset to the provided data: {e}"
        # Compare index values to the survey index (as included in self.cluster_values) after applying subsets to data
        missing_survey = ~data.index.isin(self.cluster_values.index)
        if any(missing_survey):
            return (
                f"The survey design is missing information for {missing_survey.sum():,} rows in the data,"
                f" as matched by the index values.  Here are the first few:\n"
                + str(data.loc[missing_survey].head())
            )
        return None

    def subset(self, bool_array: pd.Series) -> None:
        """
        Subset the SurveyDesignSpec (in-place) to look only at a subpopulation.

        Example:
            design.subset(data['age'] > 18)
        """
        # Confirm it is a boolean Series
        if type(bool_array) != pd.Series:
            raise ValueError(
                "SurveyDesignSpec.subset takes a boolean pandas Series object"
            )
        elif bool_array.dtype != bool:
            raise ValueError(
                "SurveyDesignSpec.subset takes a boolean pandas Series object"
            )

        # Try to subset, raising any indexing error
        try:
            # Update subset_array for tracking
            self.subset_array = self.subset_array & bool_array
            self.subset_count += 1
        except IndexingError:
            raise ValueError(
                "The boolean array passed to `subset` could not be used:"
                " the index is incompatible with the survey design"
            )

    def get_survey_design(self, regression_variable, complete_case_idx):
        """
        Get a survey design based on the SurveyDesignSpec, but specific to the rv and data

        Parameters
        ----------
        regression_variable: str
            Name of the regression variable, used to match the weight if needed
        complete_case_idx: pd.Index
            Index of the data being analyzed.  Either the same as the existing design arrays
            or smaller due to dropping rows with NA values during regression

        Returns
        -------
        sd: SurveyDesign

        """
        # Get parameters using the built-in subset
        has_strata, strata_values = (
            self.has_strata,
            self.strata_values.loc[self.subset_array],
        )
        has_cluster, cluster_values = (
            self.has_cluster,
            self.cluster_values.loc[self.subset_array],
        )
        has_weights, weight_name, weight_values = self.get_weights(regression_variable)
        # GITHUB ISSUE #118: Function self.get_weights(regression_variable) return None
        if not has_weights:
            has_weights, weight_values = (
                False,
                self.weight_values.loc[self.subset_array],
            )

        # Filter out any incomplete cases
        strata_values = strata_values.loc[complete_case_idx]
        cluster_values = cluster_values.loc[complete_case_idx]
        weight_values = weight_values.loc[complete_case_idx]

        # Initialize Survey Design
        sd = SurveyDesign(
            has_strata=has_strata,
            strat=strata_values,
            n_strat=self.n_strat,
            has_cluster=has_cluster,
            clust=cluster_values,
            n_clust=self.n_clust,
            has_weights=has_weights,
            weights=weight_values,
            has_fpc=self.has_fpc,
            fpc=self.fpc_values,
            single_cluster=self.single_cluster,
            clust_per_strat=self.clust_per_strat,
            strat_for_clust=self.strat_for_clust,
        )

        return sd


class SurveyDesign(object):
    """
    Holds the same kind of data as SurveyDesignSpec, but specific to a single regression variable:
        - Only one weight value
        - Values subset to match complete cases in the data
    """

    def __init__(
        self,
        has_strata,
        strat,
        n_strat,
        has_cluster,
        clust,
        n_clust,
        has_weights,
        weights,
        has_fpc,
        fpc,
        single_cluster,
        clust_per_strat,
        strat_for_clust,
    ):

        # Store values
        self.has_strata = has_strata
        self.strat = strat
        self.n_strat = n_strat
        self.has_cluster = has_cluster
        self.clust = clust
        self.n_clust = n_clust
        self.has_weights = has_weights
        self.weights = weights
        self.has_fpc = has_fpc
        self.fpc = fpc
        self.single_cluster = single_cluster
        self.clust_per_strat = clust_per_strat
        self.strat_for_clust = strat_for_clust

        if self.has_strata:
            self.n_strat = len(self.strat.unique())
        if self.has_cluster:
            self.n_clust = len(self.clust.unique())

    def __str__(self):
        """
        The __str__ method for our data
        """
        summary_list = [
            "Number of observations: ",
            str(len(self.strat)),
            "Sum of weights: ",
            str(self.weights.sum()),
            "Number of strata: ",
            str(self.n_strat),
            "Clusters per stratum: ",
            str(self.clust_per_strat),
        ]

        return "\n".join(summary_list)

    def get_jackknife_rep_weights(self, dropped_clust):
        """
        Computes 'delete 1' jackknife replicate weights

        Parameters
        ----------
        dropped_clust : string
            Which cluster to leave out when computing 'delete 1' jackknife replicate weights

        Returns
        -------
        w : ndarray
            Augmented weight
        """
        # get stratum that the cluster belongs in
        s = self.strat_for_clust[dropped_clust]
        nh = self.clust_per_strat[s]
        w = self.weights.copy()
        # all weights within the stratum are modified
        w[self.strat == s] *= nh / float(nh - 1)
        # but if you're within the cluster to be removed, set as 0
        w[self.clust == dropped_clust] = 0
        return w

    def get_dof(self, X):
        """
        Calculate degrees of freedom based on a subset of the design

        Parameters
        ----------
        X : pd.DataFrame
            Input data used in the calculation, possibly fewer rows than exist in the design

        Returns
        -------
        int
            Degrees of freedom
        """
        # num of clusters minus num of strata minus (num of predictors - 1)
        if self.has_cluster and self.has_strata:
            return self.n_clust - self.n_strat - (X.shape[1] - 1)
        elif self.has_cluster and not self.has_strata:
            return self.n_clust - 1 - (X.shape[1] - 1)
        elif not self.has_cluster and self.has_strata:
            return X.shape[0] - self.n_strat - (X.shape[1] - 1)
        else:
            return X.shape[0] - (X.shape[1]) - 1
