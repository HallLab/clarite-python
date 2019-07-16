import click

from .custom_types import ClariteDataParamType, ClariteEwasResultParamType

# File IO
INPUT_FILE = click.Path(exists=True, file_okay=True, dir_okay=False, readable=True)
OUTPUT_FILE = click.Path(file_okay=True, dir_okay=False, writable=True)

# Frequently used output parameters
arg_output = click.argument('output', type=OUTPUT_FILE)
option_output = click.option('--output', type=OUTPUT_FILE, default=None,
                             help='Output name.  By default the input name is used (the original data is overwritten).')

# Standard datatypes that use multiple files together
CLARITE_DATA = ClariteDataParamType()  # Instantiate it to use as a type in arguments/options
arg_data = click.argument('data', type=CLARITE_DATA)

# Tuple of dataset name, ewas df
EWAS_RESULT = ClariteEwasResultParamType()


# Skip/Only handling
def process_skip_only(ctx, param, value):
    if len(value) == 0:
        return None
    else:
        return list(value)


option_skip = click.option('-s', '--skip', type=click.STRING, multiple=True, callback=process_skip_only,
                           help="variables to skip")
option_only = click.option('-o', '--only', type=click.STRING, multiple=True, callback=process_skip_only,
                           help="variables to process, skipping all others")
