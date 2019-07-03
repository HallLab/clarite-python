import click

input_file = click.Path(exists=True, file_okay=True, dir_okay=False, readable=True)
output_file = click.Path(file_okay=True, dir_okay=False, writable=True)
