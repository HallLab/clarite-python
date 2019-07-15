from pathlib import Path
import json
from typing import Optional

import click
import pandas as pd
import numpy as np


class ClariteDataParamType(click.ParamType):
    name = "clarite-data"

    def convert(self, value, param, ctx):
        if param is None:
            return None
        try:
            if ctx is not None:
                dtypes_filename = ctx.params.get('dtypes', None)
                output = ctx.params.get('output', None)
            else:
                dtypes_filename = None
                output = None
            data = ClariteData(name=value, output=output, df=None)  # df = None b/c it will be loaded from file
            data.load_data(dtypes_filename=dtypes_filename)
            return data
        except ValueError as e:
            self.fail(f"Failed to read {value} as a CLARITE dataset."
                      f"Has it been converted using an io function?"
                      f"\n{e}",
                      param,
                      ctx)


class ClariteData:
    """
    This class manages io of various files related to the 'data' parameter in the CLI
    """
    def __init__(self,
                 name: str,
                 output: Optional[str] = None,
                 df: Optional[pd.DataFrame] = None):
        """
        Either initialize with pre-loaded data in an io function (passing df) or just a name, then call load_data
        """
        self.name = name
        self.output = output
        self.df = df
        self.dtypes = self.get_dtypes()  # Load dtypes if a df was passed- otherwise gets set to None

    def describe(self):
        """Describe the df for logging"""
        if self.df is None:
            return "empty DataFrame"
        else:
            return f"{len(self.df):,} observations of {len(self.df.columns):,} variables"

    def load_data(self, dtypes_filename: str = None):
        """
        Load:
            name.txt (tsv file) into a self.df
            name.dtypes (json file) into self.dtypes
        """
        # Load dtypes from file (default or specified file)
        if dtypes_filename is None:
            dtypes_filename = self.name + ".dtypes"
        dtypes_file = Path(dtypes_filename)
        if not dtypes_file.exists():
            raise ValueError(f"Could not read '{dtypes_filename}'")
        else:
            with dtypes_file.open('r') as f:
                try:
                    self.dtypes = json.load(f)
                except json.JSONDecodeError as e:
                    raise ValueError(f"'{dtypes_filename}' was not a valid dtypes file: {e}")

        # Load Data
        data_filename = Path(self.name + ".txt")
        data_file = Path(data_filename)
        if not data_file.exists():
            raise ValueError(f"Could not read '{data_filename}'")
        else:
            self.df = pd.read_csv(data_file, sep="\t")

    def save_data(self):
        """
        Save data and associated files
        """
        # Skip saving if there is no data
        if len(self.df) == 0:
            click.echo(click.style(f"No variables in {self.name}: {self.output} was not created.", fg='yellow'))
        # Refresh dtypes in case the df was modified
        self.dtypes = self.get_dtypes()
        # Where to save
        if self.output is None:
            self.output = self.name
        output_filename = self.output + ".txt"
        output_file = Path(output_filename)
        # Prepare to save dtypes
        dtypes_filename = self.output + ".dtypes"
        dtypes_file = Path(dtypes_filename)

        # Save data
        self.df.to_csv(output_file, sep="\t")
        # Save dtypes
        with dtypes_file.open('w') as f:
            json.dump(self.dtypes, f)

        # Log
        click.echo(click.style(f"Done: Saved {self.describe()} to {self.output}", fg='green'))

    def get_dtypes(self):
        """
        Convert dtypes of the DataFrame to a dictionary format
        Examples:
        binary: {'female': {'type': 'category', 'categories': [0, 1], 'ordered': False}}
        categorical: {'CALCIUM_Unknown': {'type': 'category', 'categories': [0.0, 0.066666666, 0.933333333], 'ordered': False}}
        continuous: {'BMXBMI': {'type': 'float64'}}
        """
        if self.df is None:
            return None
        dtypes = {variable_name: {'type': str(dtype)} if str(dtype) != 'category'
                  else {'type': str(dtype), 'categories': list(dtype.categories.values.tolist()), 'ordered': dtype.ordered}
                  for variable_name, dtype in self.df.dtypes.iteritems()}
        return dtypes

    def set_dtypes(self, dtypes):
        """
        Set the dtypes of a dataframe according to a dtypes dictionary (in-place)
        """
        # Validate
        missing_types = set(list(self.df)) - set(dtypes.keys())
        extra_dtypes = set(dtypes.keys()) - set(list(self.df))
        if len(missing_types) > 0:
            raise ValueError(f"Dtypes file is missing some values: {', '.join(missing_types)}")
        if len(extra_dtypes) > 0:
            raise ValueError(f"Dtypes file has types for variables not found in the data: {', '.join(extra_dtypes)}")

        for col in list(self.df):
            typeinfo = dtypes[col]
            newtype = typeinfo['type']
            if typeinfo['type'] == 'category':
                newtype = pd.CategoricalDtype(categories=np.array(typeinfo['categories']), ordered=typeinfo['ordered'])
            self.df[col] = self.df[col].astype(newtype)
