import numpy as np
from scipy import stats
from scipy.optimize import brentq as root

import re

from clarite.modules.analyze.utils import statsmodels_var_regex


def regTermTest(full_model, restricted_model, ddf, X_names, var_name):
    """
    Performs a LRT for two weighted regression models- one full and one restricted (which nests inside the larger model).
    This is a limited adaption of the same function in the R 'survey' library.
    """
    # Get chisq
    # Note: These are not the correct deviance values, which need to account for weights
    chisq = restricted_model.result.deviance - full_model.result.deviance

    # Get Misspec
    idx = [n for n in X_names if re.match(statsmodels_var_regex(var_name), n)]
    V = full_model.vcov.loc[idx, idx]
    V0 = (full_model.result.cov_params() / full_model.result.scale).loc[idx, idx]
    misspec = np.linalg.eig(np.matmul(np.linalg.pinv(V0), V))[0]

    # Calculate p
    p = _pFsum(x=chisq, df=np.ones(len(misspec)), a=misspec, ddf=ddf)

    return p


def _pFsum(x, df, a, ddf):
    tr = a.mean()
    tr2 = (a ** 2).mean() / tr ** 2
    scale = tr * tr2
    ndf = len(a) / tr2

    rval = stats.f.sf(x / ndf / scale, ndf, ddf)

    # Look for saddlepoint
    a = np.append(a, -x / ddf)
    df = np.append(df, ddf)
    if any(df > 1):
        a = np.repeat(a, df.tolist(), axis=0)
    s = _saddle(x=0, lam=a)
    if ~np.isnan(s):
        rval = s

    return rval


def _saddle(x, lam):
    """
    This function is used within the pFsum function
    """
    d = max(lam)
    lam = lam / d
    x = x / d

    def k0(z):
        return -1 * np.log(1 - 2 * z * lam).sum() / 2

    def kprime0(z):
        return (lam / (1 - 2 * z * lam)).sum()

    def kpprime0(z):
        return 2 * (lam ** 2 / (1 - 2 * z * lam) ** 2).sum()

    if any(lam < 0):
        lam_min = (1 / (2 * lam[lam < 0])).max() * 0.99999
    elif x > lam.sum():
        lam_min = -0.01
    else:
        lam_min = -1 * len(lam) / (2 * x)

    lam_max = (1 / (2 * lam[lam > 0])).min() * 0.99999

    hatzeta = root(lambda z: kprime0(z) - x, a=lam_min, b=lam_max)

    sign = 1 if hatzeta > 0 else -1
    w = sign * np.sqrt(2 * (hatzeta * x - k0(hatzeta)))
    v = hatzeta * np.sqrt(kpprime0(hatzeta))

    if np.abs(hatzeta) < 1e-04:
        return np.nan
    else:
        return stats.norm.sf(w + np.log(v / w) / w)
