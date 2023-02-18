import numpy as np
import pandas as pd
from statsmodels.genmod import generalized_linear_model


class SurveyModel(object):
    """

    Parameters
    -------
    design : Instance of class SurveyDesign
    model_class : Instance of class GLM
    init_args : Dictionary of arguments
        when initializing the model
    fit_args : Dictionary of arguments
        when fitting the model

    Attributes
    ----------
    design : Instance of class SurveyDesign
    model : Instance of class GLM
    init_args : Dictionary of arguments
        when initializing the model
    fit_args : Dictionary of arguments
        when fitting the model
    params : (p, ) array
        Array of coefficients of model
    vcov : (p, p) array
        Covariance matrix
    stderr : (p, ) array
        Standard error of cofficients
    """

    def __init__(self, design, model_class, init_args={}, fit_args={}):
        # TODO: Take in original (full) data which may have more observations than the fitted data (due to NA dropouts)
        self.design = design
        self.model = model_class
        self.init_args = dict(init_args)
        self.fit_args = dict(fit_args)

        if self.model is generalized_linear_model.GLM:
            self.glm_flag = True
        else:
            self.glm_flag = False

    def _stata_linearization_vcov(self, X: pd.DataFrame, y: pd.DataFrame):
        """
        Get the linearized covariance matrix using STATA's methodology

        Parameters
        ----------
        X : pd.DataFrame
            A n x p dataframe where 'n' is the number of observations and 'p'
            is the number of regressors.
        y : pd.DataFrame
            1-column dataframe of the response variable

        Returns
        -------
        vcov : (p,p) array
            The covariance matrix

        Notes
        -----
        This uses a 'sandwich' method, where three matrices are multiplied
        together. The outer parts of the 'sandwich' is the inverse of the
        negative hessian. The inside is the design-based variance of the
        estimation of a total, ie np.dot(weights, observations) where
        'observations' is the derivative of the log pseudolikelihood w.r.t
        theta (X*Beta) multiplied by the data

        Reference
        ---------
        http://www.stata.com/manuals13/svyvarianceestimation.pdf
        """

        # The term "variables" used below means the variables in the specific regression, such as:
        #  Intercept
        #  One value for each continuous and binary variable
        #  C-1 values for each categorical where C is the number of categories

        # Get hessian and inverse hessian
        # Rows and Columns are the variables
        # Both are symmetric across the diagonal
        hessian = pd.DataFrame(
            self.initialized_model.hessian(self.params, observed=True),
            index=self.params.index,
            columns=self.params.index,
        )
        hess_inv = pd.DataFrame(
            np.linalg.inv(hessian.values),
            index=self.params.index,
            columns=self.params.index,
        )

        # Get the Jacobian/Gradient of log-likelihood evaluated at params for each observation
        # Statsmodels stores this in the model
        # This matrix has one row for each observation (index = 'ID') and a column for each variable
        d_hat = pd.DataFrame(
            self.initialized_model.score_obs(self.params),
            index=X.index,
            columns=self.params.index,
        )

        # Add strat and clust information to the d_hat index (this just modifies the index labels)
        d_hat = pd.merge(
            d_hat, self.design.strat, left_index=True, right_index=True
        ).set_index("strat", append=True)
        d_hat = pd.merge(
            d_hat, self.design.clust, left_index=True, right_index=True
        ).set_index("clust", append=True)

        # Get sum of d_hat within each cluster (rows = clusters, columns = variables)
        jdata = d_hat.groupby(axis=0, level="clust", group_keys=False).apply(sum)

        if self.design.has_strata:
            # Add strata label to jdata (just updates the index labels)
            jdata = pd.merge(
                jdata,
                self.design.strat_for_clust.loc[jdata.index].rename("strat"),
                left_index=True,
                right_index=True,
            ).set_index("strat", append=True)

            # Center each stratum in jdata (which is one or more rows, each corresponding to individual clusters)

            def center_strata(data, single_cluster_setting, pop_mean):
                """
                Center strata around zero, logging any single-cluster strata
                """
                if len(data) == 1 and single_cluster_setting == "adjust":
                    # Subtract the population mean from the single-cluster strata
                    return data - pop_mean
                else:
                    # Substract the mean across clusters in this strata
                    # This results in a value of 0 for the strata when there is only 1 cluster
                    return data - data.mean()

            jdata = jdata.groupby(axis=0, level="strat", group_keys=False).apply(
                lambda g: center_strata(g, self.design.single_cluster, d_hat.mean())
            )

            # Scale after centering, if required
            # TODO: Scale may be different if some strat/clusters drop?
            if self.design.single_cluster == "average":
                single_cluster_scale = np.sqrt(
                    self.design.n_strat
                    / (self.design.n_strat - sum(self.design.clust_per_strat == 1))
                )
                jdata *= single_cluster_scale

        # nh is one row per cluster, listing the number of clusters in the strata that the cluster belongs to
        # Note that this includes strata in the original design, before any subsetting of any kind
        nh = self.design.clust_per_strat.loc[self.design.strat_for_clust].astype(
            np.float64
        )
        mh = np.sqrt(nh / (nh - 1))
        mh[
            mh == np.inf
        ] = 1  # Replace infinity with 1.0 (due to single cluster where nh==1)
        fh = np.sqrt(
            1 - self.design.fpc
        )  # self.design.fpc has fpc value for each cluster (one row per cluster)
        jdata *= (
            fh.values[:, None] * mh.values[:, None]
        )  # This is each column in jdata being multiplied by an array

        v_hat = np.dot(jdata.T, jdata)  # variables X variables
        vcov = np.dot(hess_inv, v_hat).dot(hess_inv.T)  # variables X variables
        # Return as a data.frame
        vcov = pd.DataFrame(vcov, index=self.params.index, columns=self.params.index)
        return vcov

    def _jackknife_vcov(self, X, y):
        # Calculate parameter values, leaving out one cluster at a time
        replicate_params = dict()
        for c in self.design.clust_names:
            w = self.design.get_jackknife_rep_weights(dropped_clust=c)
            init_args = {k: v for k, v in self.init_args.items()}
            # Add weights parameter, indexing by 'X' to remove rows that dropped b/c they were not complete cases
            if self.glm_flag:
                init_args["freq_weights"] = w.loc[X.index]
            else:
                init_args["weights"] = w.loc[X.index]
            replicate_params[c] = self._calc_params(y, X, init_args, self.fit_args)

        # Collect data into a dataframe and center it
        replicate_params = pd.DataFrame.from_dict(
            replicate_params, orient="index", columns=self.params.index
        )
        replicate_params.index.name = "clust"
        self.replicate_params = replicate_params - self.params

        nh = self.design.clust_per_strat.loc[self.design.strat_for_clust].astype(
            np.float64
        )
        mh = np.sqrt((nh - 1) / nh)
        mh[
            mh == np.inf
        ] = 1  # Replace infinity with 1.0 (due to single cluster where nh==1)
        fh = np.sqrt(1 - self.design.fpc)
        self.replicate_params *= mh[:, None] * fh[:, None]

        vcov = np.dot(self.replicate_params.T, self.replicate_params)

        # Return as a data.frame
        vcov = pd.DataFrame(vcov, index=self.params.index, columns=self.params.index)
        return vcov

    def fit(self, y, X, center_by="est"):

        assert y.index.equals(X.index)
        self.center_by = center_by

        # Get weights
        # Will be an array of ones if not provided
        # Need to index by X since patsy dropped incomplete cases when creating X and the weights array includes them
        weights = self.design.weights.loc[X.index]

        # Use weights in the regression
        if self.glm_flag:
            self.init_args["freq_weights"] = weights
        else:
            self.init_args["weights"] = weights
        self.params = self._get_params(y, X)

        if self.design.has_strata or self.design.has_cluster or self.design.has_weights:
            # Calculate stderr based on covariance
            self.vcov = self._stata_linearization_vcov(X, y)
            if self.vcov.ndim == 2:
                self.stderr = np.sqrt(np.diag(self.vcov))
            else:
                self.stderr = np.sqrt(self.vcov)
        else:
            # Use the stderr calculated in the original weighted regression
            self.stderr = self.result.bse

    def _get_params(self, y, X):
        self.initialized_model = self.model(y, X, **self.init_args)
        self.result = self.initialized_model.fit(**self.fit_args)
        return self.result.params

    def _calc_params(self, y, X, init_args, fit_args):
        """Like _get_params, but doesn't modify self.  Jackknife calculates params many times, but must keep original values."""
        initialized_model = self.model(y, X, **init_args)
        result = initialized_model.fit(**fit_args)
        return result.params
