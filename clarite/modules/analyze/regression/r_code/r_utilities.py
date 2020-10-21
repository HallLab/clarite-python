import numpy as np
import pandas as pd
from rpy2 import robjects as ro


def ewasresult2py(r_result):
    """
    Convert EWAS results from R into a pandas DataFrame.
    """
    df = ro.conversion.rpy2py(r_result)
    df = df.rename(columns={"outcome": "Outcome", "weight": "Weight", "pval": "pvalue"})
    df["Converged"] = df["Converged"].astype(bool)
    df = df.replace(
        "None", np.nan
    )  # The EWAS R function specifically returns 'None' instead of NA
    df = df.set_index(["Variable", "Outcome"])
    return df


def df_pandas2r(data: pd.DataFrame) -> ro.vectors.DataFrame:
    """Convert pandas dataframe to R DataFrame, making sure categoricals/factors are correctly preserved"""
    # Make sure categories are strings
    for col in data.columns:
        if data[col].dtype.name == "category":
            data[col] = data[col].cat.rename_categories(
                [str(c) for c in data[col].cat.categories]
            )
    data = ro.vectors.DataFrame(data)
    return data
