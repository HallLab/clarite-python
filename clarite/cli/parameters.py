import click

input_file = click.Path(exists=True, file_okay=True, dir_okay=False, readable=True)
output_file = click.Path(file_okay=True, dir_okay=False, writable=True)

skip = click.option('--skip', '-s', type=click.STRING, multiple=True, help="variables to skip")
only = click.option('--only', '-o', type=click.STRING, multiple=True, help="variables to process, skipping all others")
