import click

from .custom_types import ClariteDataParamType

# File IO
INPUT_FILE = click.Path(exists=True, file_okay=True, dir_okay=False, readable=True)
OUTPUT_FILE = click.Path(file_okay=True, dir_okay=False, writable=True)

# Standard datatype that uses multiple files together
CLARITE_DATA = ClariteDataParamType()  # Instantiate it to use as a type in arguments/options
clarite_data_arg = click.argument('data', type=CLARITE_DATA)


# Skip/Only handling
def process_skip_only(ctx, param, value):
    if len(value) == 0:
        return None
    else:
        return list(value)


skip = click.option('-s', '--skip', type=click.STRING, multiple=True, callback=process_skip_only,
                    help="variables to skip")
only = click.option('-o', '--only', type=click.STRING, multiple=True, callback=process_skip_only,
                    help="variables to process, skipping all others")
