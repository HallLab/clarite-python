import numpy as np
import pandas as pd
from statsmodels.genmod import generalized_linear_model


class SurveyModel(object):
    """

    Parameters
    -------
    design : Instance of class SurveyDesign
    model_class : Instance of class GLM
    cov_method : str
        Method for calculating covariance
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
    def __init__(self, design, model_class, cov_method='stata', init_args={}, fit_args={}):
        # TODO: Take in original (full) data which may have more observations than the fitted data (due to NA dropouts)
        self.design = design
        self.model = model_class
        self.cov_method = cov_method
        self.init_args = dict(init_args)
        self.fit_args = dict(fit_args)

        if self.model is generalized_linear_model.GLM:
            self.glm_flag = True
        else:
            self.glm_flag = False

    def _stata_linearization_vcov(self, X, y):
        """
        Get the linearized covariance matrix using STATA's methodology

        Parameters
        ----------
        X : array-like
            A n x p array where 'n' is the number of observations and 'p'
            is the number of regressors.
        y : array-like
            1d array of the response variable

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
        # Input data indicies must match design indicies
        design_index = self.design.strat.index  # May be all ones, but it always exists
        assert X.index.equals(design_index)
        assert y.index.equals(design_index)

        # Get hessian
        hessian = pd.DataFrame(self.initialized_model.hessian(self.params, observed=True),
                               index=self.params.index,
                               columns=self.params.index)
        hess_inv = pd.DataFrame(np.linalg.inv(hessian.values),
                                index=self.params.index,
                                columns=self.params.index)

        # Get the first derivative of the loglikelihood function evaluated at params for each observation
        d_hat = pd.DataFrame(self.initialized_model.score_obs(self.params),
                             index=self.design.strat.index,
                             columns=self.params.index)

        # Add strat and clust information to the d_hat index
        d_hat = pd.merge(d_hat, self.design.strat, left_index=True, right_index=True).set_index('strat', append=True)
        d_hat = pd.merge(d_hat, self.design.clust, left_index=True, right_index=True).set_index('clust', append=True)

        # Get sum of d_hat within each cluster
        jdata = d_hat.groupby(axis=0, level='clust').apply(sum)

        # Add strata label to jdata
        jdata = pd.merge(jdata, self.design.strat_for_clust.loc[jdata.index].rename('strat'), left_index=True, right_index=True)\
                  .set_index('strat', append=True)

        def center_strata(data, single_cluster, pop_mean):
            if len(data) > 1 or single_cluster == 'average' or single_cluster == 'certainty':
                # Substract the mean across clusters
                # This results in a value of 0 for the strata when there is only 1 cluster
                return data - data.mean()
            elif len(data) == 1 and single_cluster == 'adjust':
                # Subtract the population mean from the single-cluster strata
                return data - pop_mean
            elif len(data) == 1:
                raise ValueError(f"Strat '{data.name}' has a single cluster. "
                                 f"Adjust the 'single_cluster' SurveyDesign parameter or reassign the cluster to avoid this error.")

        # Center each stratum
        single_cluster = self.design.single_cluster  # 'average', 'certainty', or 'adjust'.  Anything else will throw an error for single-cluster-strata
        jdata = jdata.groupby(axis=0, level='strat').apply(lambda g: center_strata(g, single_cluster, d_hat.mean()))

        # Scale after centering, if required
        if single_cluster == 'average':
            single_cluster_scale = np.sqrt(self.design.n_strat / (self.design.n_strat - sum(self.design.clust_per_strat == 1)))
            print(single_cluster_scale)
            print(sum(self.design.clust_per_strat == 1))
            jdata *= single_cluster_scale

        nh = self.design.clust_per_strat.loc[self.design.strat_for_clust].astype(np.float64)
        mh = np.sqrt(nh / (nh-1))
        mh[mh == np.inf] = 1  # Replace infinity with 1.0 (due to single cluster where nh==1)
        fh = np.sqrt(1 - self.design.fpc)
        jdata *= (fh[:, None] * mh[:, None])
        v_hat = np.dot(jdata.T, jdata)
        vcov = np.dot(hess_inv, v_hat).dot(hess_inv.T)
        # Return as a data.frame
        vcov = pd.DataFrame(vcov, index=self.params.index, columns=self.params.index)
        return vcov

    def _jackknife_vcov(self, X, y):
        # Calculate parameter values, leaving out one cluster at a time
        replicate_params = dict()
        for c in self.design.clust_names:
            w = self.design.get_jackknife_rep_weights(dropped_clust=c)
            init_args = {k: v for k, v in self.init_args.items()}
            if self.glm_flag:
                init_args["freq_weights"] = w
            else:
                init_args["weights"] = w
            replicate_params[c] = self._calc_params(y, X, init_args, self.fit_args)

        # Collect data into a dataframe and center it
        replicate_params = pd.DataFrame.from_dict(replicate_params, orient='index', columns=self.params.index)
        replicate_params.index.name = 'clust'
        self.replicate_params = replicate_params - self.params

        nh = self.design.clust_per_strat.loc[self.design.strat_for_clust].astype(np.float64)
        mh = np.sqrt((nh - 1) / nh)
        mh[mh == np.inf] = 1  # Replace infinity with 1.0 (due to single cluster where nh==1)
        fh = np.sqrt(1 - self.design.fpc)
        self.replicate_params *= (mh[:, None] * fh[:, None])

        vcov = np.dot(self.replicate_params.T, self.replicate_params)

        # Return as a data.frame
        vcov = pd.DataFrame(vcov, index=self.params.index, columns=self.params.index)
        return vcov

    def fit(self, y, X, center_by='est'):

        assert y.index.equals(X.index)
        self.center_by = center_by

        # Get weights (will be an array of ones if not provided)
        weights = self.design.weights
        # Normalize weights
        weights = weights/weights.mean()
        # Use weights in the regression
        if self.glm_flag:
            self.init_args["freq_weights"] = weights
        else:
            self.init_args["weights"] = weights
        self.params = self._get_params(y, X)

        if self.design.has_strata or self.design.has_clusters or self.design.has_weights:
            # Calculate stderr based on covariance
            if self.cov_method == 'jackknife':
                self.vcov = self._jackknife_vcov(X, y)
            elif self.cov_method == 'stata':
                self.vcov = self._stata_linearization_vcov(X, y)
            else:
                return ValueError(f"cov_method '{self.cov_method}' is not supported")
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
