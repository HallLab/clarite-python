import click

input_file = click.Path(exists=True, file_okay=True, dir_okay=False, readable=True)
output_file = click.Path(file_okay=True, dir_okay=False, writable=True)

skip = click.option('-s', '--skip', type=click.STRING, multiple=True, help="variables to skip")
only = click.option('-o', '--only', type=click.STRING, multiple=True, help="variables to process, skipping all others")
