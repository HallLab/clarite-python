from typing import Optional, List

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq as root


def _validate_skip_only(columns, skip: Optional[List[str]] = None, only: Optional[List[str]] = None):
    """Validate use of the 'skip' and 'only' parameters, returning a valid list of columns to filter"""
    if skip is not None and only is not None:
        raise ValueError("It isn't possible to specify 'skip' and 'only' at the same time.")
    elif skip is not None and only is None:
        invalid_cols = set(skip) - set(columns)
        if len(invalid_cols) > 0:
            raise ValueError(f"Invalid columns passed to 'skip': {', '.join(invalid_cols)}")
        columns = [c for c in columns if c not in set(skip)]
    elif skip is None and only is not None:
        invalid_cols = set(only) - set(columns)
        if len(invalid_cols) > 0:
            raise ValueError(f"Invalid columns passed to 'only': {', '.join(invalid_cols)}")
        columns = [c for c in columns if c in set(only)]

    if len(columns) == 0:
        raise ValueError("No columns available for filtering")

    return columns


def make_bin(df: pd.DataFrame):
    """
    Validate and type a dataframe of binary variables

    Checks that each variable has at most 2 values and converts the type to pd.Categorical

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame to be processed

    Returns
    -------
    df: pd.DataFrame
        DataFrame with the same data but validated and converted to categorical types

    Examples
    --------
    >>> df = clarite.make_bin(df)
    Processed 32 binary variables with 4,321 observations
    """
    # TODO: add further validation
    df = df.astype('category')
    print(f"Processed {len(df.columns):,} binary variables with {len(df):,} observations")
    return df


def make_categorical(df: pd.DataFrame):
    """
    Validate and type a dataframe of categorical variables

    Converts the type to pd.Categorical

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame to be processed

    Returns
    -------
    df: pd.DataFrame
        DataFrame with the same data but validated and converted to categorical types

    Examples
    --------
    >>> df = clarite.make_categorical(df)
    Processed 12 categorical variables with 4,321 observations
    """
    # TODO: add further validation
    df = df.astype('category')
    print(f"Processed {len(df.columns):,} categorical variables with {len(df):,} observations")
    return df


def make_continuous(df: pd.DataFrame):
    """
    Validate and type a dataframe of continuous variables

    Converts the type to numeric

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame to be processed

    Returns
    -------
    df: pd.DataFrame
        DataFrame with the same data but validated and converted to numeric types

    Examples
    --------
    >>> df = clarite.make_continuous(df)
    Processed 128 continuous variables with 4,321 observations
    """
    # TODO: add further validation
    df = df.apply(pd.to_numeric)
    print(f"Processed {len(df.columns):,} continuous variables with {len(df):,} observations")
    return df


def regTermTest(full_model, restricted_model, ddf, X_names, var_name):
    """
    Performs a LRT for two weighted regression models- one full and one restricted (which nests inside the larger model).
    This is a limited adaption of the same function in the R 'survey' library.
    """
    # Get chisq
    chisq = restricted_model.result.deviance - full_model.result.deviance

    # Get Misspec
    idx = [f"C({var_name})" in n for n in X_names]
    V = full_model.vcov.loc[idx, idx]
    V0 = (full_model.result.cov_params()/full_model.result.scale).loc[idx, idx]
    misspec = np.linalg.eig(np.matmul(np.linalg.pinv(V0), V))[0]

    # Calculate p
    p = _pFsum(x=chisq, df=np.ones(len(misspec)), a=misspec, ddf=ddf)

    return p


def _pFsum(x, df, a, ddf):
    tr = a.mean()
    tr2 = (a**2).mean()/tr**2
    scale = tr * tr2
    ndf = len(a)/tr2

    rval = stats.f.sf(x/ndf/scale, ndf, ddf)

    # Look for saddlepoint
    a = np.append(a, -x/ddf)
    df = np.append(df, ddf)
    if(any(df > 1)):
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
    x = x/d

    def k0(z): return -1 * np.log(1-2*z*lam).sum()/2
    def kprime0(z): return (lam/(1-2*z*lam)).sum()
    def kpprime0(z): return 2*(lam**2/(1-2*z*lam)**2).sum()

    if any(lam < 0):
        lam_min = (1/(2*lam[lam < 0])).max() * 0.99999
    elif x > lam.sum():
        lam_min = -0.01
    else:
        lam_min = -1 * len(lam)/(2*x)

    lam_max = (1/(2*lam[lam > 0])).min() * 0.99999

    hatzeta = root(lambda z: kprime0(z) - x, a=lam_min, b=lam_max)

    sign = 1 if hatzeta > 0 else -1
    w = sign * np.sqrt(2*(hatzeta*x-k0(hatzeta)))
    v = hatzeta * np.sqrt(kpprime0(hatzeta))

    if np.abs(hatzeta) < 1e-04:
        return np.nan
    else:
        return 1-stats.norm.cdf(w + np.log(v/w)/w)
