import numpy as np
import pandas as pd
from rpy2.rinterface import NALogicalType


def ewasresult2py(r_result):
    """
    Convert EWAS results from R into a pandas DataFrame.
    This can likely be replaced with rpy2 functionality when it is fixed for pandas 1.0.
    """
    result = pd.DataFrame.from_dict({
        'Variable': np.asarray(r_result.rx2("Variable"), dtype="object"),
        'Phenotype': np.asarray(r_result.rx2("phenotype"), dtype="object"),
        'N': np.asarray(r_result.rx2("N")),
        'Converged': np.asarray(r_result.rx2("Converged"), dtype=bool),
        'Beta': np.asarray([np.NaN if type(v) == NALogicalType else v for v in r_result.rx2("Beta")], dtype=float),
        'SE': np.asarray([np.NaN if type(v) == NALogicalType else v for v in r_result.rx2("SE")], dtype=float),
        'Variable_pvalue': np.asarray([np.NaN if type(v) == NALogicalType else v for v in r_result.rx2("Variable_pvalue")], dtype=float),
        'LRT_pvalue': np.asarray([np.NaN if type(v) == NALogicalType else v for v in r_result.rx2("LRT_pvalue")], dtype=float),
        'Diff_AIC': np.asarray([np.NaN if type(v) == NALogicalType else v for v in r_result.rx2("Diff_AIC")], dtype=float),
        'pvalue': np.asarray([np.NaN if type(v) == NALogicalType else v for v in r_result.rx2("pval")], dtype=float),
        'weight': np.asarray([np.NaN if type(v) == NALogicalType else v for v in r_result.rx2("weight")], dtype=object),
    })
    result = result.set_index(['Variable', 'Phenotype'])
    return result
