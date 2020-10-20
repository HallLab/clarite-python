from pathlib import Path
import click

from .custom_types import ClariteDataParamType, ClariteEwasResultParamType

# File IO
INPUT_FILE = click.Path(exists=True, file_okay=True, dir_okay=False, readable=True)
OUTPUT_FILE = click.Path(file_okay=True, dir_okay=False, writable=True)

# Frequently used output parameters
arg_output = click.argument("output", type=OUTPUT_FILE)

# Standard datatypes that use multiple files together
CLARITE_DATA = (
    ClariteDataParamType()
)  # Instantiate it to use as a type in arguments/options
arg_data = click.argument("data", type=CLARITE_DATA)

# Tuple of dataset name, ewas df
EWAS_RESULT = ClariteEwasResultParamType()


# Skip/Only handling
def process_skip_only(ctx, param, value):
    # Value is a tuple of passed inputs (for each --skip or --only)
    option_name = param.name
    if len(value) == 0:
        # No values passed
        return None
    else:
        result = []
        as_files = {}
        as_strings = 0
        for v in value:
            p = Path(v)
            if p.exists() and p.is_file():
                # Try loading a list of variables
                with p.open("r") as f:
                    file_result = [v.strip() for v in f.readlines()]
                    file_result = [
                        v for v in file_result if v != ""
                    ]  # skip blank lines
                    as_files[v] = len(file_result)
                    result += file_result
            else:
                # Assume this is the name of a variable
                as_strings += 1
                result.append(v)
        click.echo("-" * 80)
        click.echo(f"--{option_name}: {as_strings} variable(s) specified directly")
        for filename, num in as_files.items():
            click.echo(f"\t{num:,} variable(s) loaded from '{filename}'")
        return result


option_skip = click.option(
    "-s",
    "--skip",
    type=click.STRING,
    multiple=True,
    callback=process_skip_only,
    help="variables to skip.  Either individual names, or a file containing one name per line.",
)
option_only = click.option(
    "-o",
    "--only",
    type=click.STRING,
    multiple=True,
    callback=process_skip_only,
    help="variables to process, skipping all others.  Either individual names, or a file containing one name per line.",
)
