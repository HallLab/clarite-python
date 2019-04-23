from typing import Optional

import pandas as pd


def load_data(filename: str,
              id_col: Optional[str] = None,
              **kwargs):
    df = pd.read_csv(filename, **kwargs)\
           .set_index("ID")
    print(f"Loaded {len(df):,} observations of {len(df.columns):,} variables")
    return df
