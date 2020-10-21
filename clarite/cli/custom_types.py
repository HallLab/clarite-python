from pathlib import Path
import json
from typing import Optional

import click
import pandas as pd
import numpy as np

from ..modules import analyze


class ClariteData:
    """
    This class manages loading various files related to the 'data' parameter in the CLI
    """

    def __init__(self, name: str, df: Optional[pd.DataFrame] = None):
        """
        Either initialize with pre-loaded data in an io function (passing df) or just a name
        """
        self.name = name
        self.df = df
        if self.df is None:
            self.load_data()
        self.dtypes = (
            self.get_dtypes()
        )  # Load dtypes if a df was passed- otherwise gets set to None

    def describe(self):
        """Describe the df for logging"""
        if self.df is None:
            return "empty DataFrame"
        else:
            return (
                f"{len(self.df):,} observations of {len(self.df.columns):,} variables"
            )

    def load_data(self):
        """
        Load:
            name.txt (tsv file) into a self.df
            name.dtypes (json file) into self.dtypes
        """
        # Load Data
        data_filename = Path(self.name + ".txt")
        data_file = Path(data_filename)
        if not data_file.exists():
            raise ValueError(f"Could not read '{data_filename}'")
        else:
            self.df = pd.read_csv(data_file, sep="\t", index_col="ID")
        # Load dtypes from file
        dtypes_filename = self.name + ".dtypes"
        dtypes_file = Path(dtypes_filename)
        if not dtypes_file.exists():
            raise ValueError(f"Could not read '{dtypes_filename}'")
        with dtypes_file.open("r") as f:
            try:
                self.dtypes = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"'{dtypes_filename}' was not a valid dtypes file: {e}"
                )
        # Update df dtypes
        self.set_dtypes(self.dtypes)

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
        dtypes = {
            variable_name: {"type": str(dtype)}
            if str(dtype) != "category"
            else {
                "type": str(dtype),
                "categories": list(dtype.categories.values.tolist()),
                "ordered": dtype.ordered,
            }
            for variable_name, dtype in self.df.dtypes.iteritems()
        }
        return dtypes

    def set_dtypes(self, dtypes):
        """
        Set the dtypes of a dataframe according to a dtypes dictionary (in-place)
        """
        # Validate
        missing_types = set(list(self.df)) - set(dtypes.keys())
        extra_dtypes = set(dtypes.keys()) - set(list(self.df))
        if len(missing_types) > 0:
            raise ValueError(
                f"Dtypes file is missing some values: {', '.join(missing_types)}"
            )
        if len(extra_dtypes) > 0:
            raise ValueError(
                f"Dtypes file has types for variables not found in the data: {', '.join(extra_dtypes)}"
            )

        for col in list(self.df):
            typeinfo = dtypes[col]
            newtype = typeinfo["type"]
            if typeinfo["type"] == "category":
                newtype = pd.CategoricalDtype(
                    categories=np.array(typeinfo["categories"]),
                    ordered=typeinfo["ordered"],
                )
            self.df[col] = self.df[col].astype(newtype)


def save_clarite_data(data: ClariteData, output: str = None):
    """
    Save CLARITE data and associated files in a standard format.
    """
    # Use the input name as output if one isn't provided
    if output is None:
        output = data.name

    # Skip saving if there is no data
    if len(data.df) == 0:
        click.echo(
            click.style(
                f"No variables to output: {output}.txt was not written.", fg="yellow"
            )
        )

    # Refresh dtypes in case the df was modified
    data.dtypes = data.get_dtypes()

    # Save data
    output_filename = output + ".txt"
    output_file = Path(output_filename)
    data.df.to_csv(output_file, sep="\t")

    # Save dtypes
    dtypes_filename = output + ".dtypes"
    dtypes_file = Path(dtypes_filename)
    with dtypes_file.open("w") as f:
        json.dump(data.dtypes, f)

    # Save Log file
    # TODO

    # Log
    click.echo(click.style(f"Done: Saved {data.describe()} to {output}", fg="green"))


def save_clarite_ewas(data: pd.DataFrame, output: str = None):
    """
    Save CLARITE EWAS result.
    """
    # Skip saving if there is no data
    if len(data) == 0:
        click.echo(
            click.style(
                f"No variables to output: {output}.txt was not written.", fg="yellow"
            )
        )

    # Save data
    output_filename = output + ".txt"
    output_file = Path(output_filename)
    data.to_csv(output_file, sep="\t")

    # Log
    click.echo(
        click.style(
            f"Done: Saved EWAS results for {len(data):,} variables to {output}",
            fg="green",
        )
    )


class ClariteDataParamType(click.ParamType):
    name = "clarite-data"

    def convert(self, value, param, ctx):
        if param is None:
            return None
        try:
            data = ClariteData(
                name=value, df=None
            )  # df = None b/c it will be loaded from file
            return data
        except ValueError as e:
            self.fail(
                f"Failed to read {value} as a CLARITE dataset. "
                f"Has it been converted using an io function?"
                f"\n\t{e}",
                param,
                ctx,
            )


class ClariteEwasResultParamType(click.ParamType):
    name = "clarite-ewas-result"

    def convert(self, value, param, ctx):
        if param is None:
            return None
        try:
            # Load data
            data = pd.read_csv(
                value + ".txt", sep="\t", index_col=["Variable", "Outcome"]
            )
            # Check columns
            cols_original = analyze.result_columns
            cols_with_corrected_pvals = (
                analyze.result_columns + analyze.corrected_pvalue_columns
            )
            if (list(data) != cols_original) & (
                list(data) != cols_with_corrected_pvals
            ):
                raise ValueError(f"{value} was not a valid EWAS result file.")
            return (value, data)  # tuple to include name
        except ValueError as e:
            self.fail(
                f"Failed to read {value}.txt as a CLARITE EWAS result dataset. "
                f"\n\t{e}",
                param,
                ctx,
            )
