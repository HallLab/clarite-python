import click

input_file = click.Path(exists=True, file_okay=True, dir_okay=False, readable=True)
output_file = click.Path(file_okay=True, dir_okay=False, writable=True)


def process_skip_only(ctx, param, value):
    if len(value) == 0:
        return None
    else:
        return list(value)


skip = click.option('-s', '--skip', type=click.STRING, multiple=True, callback=process_skip_only,
                    help="variables to skip")
only = click.option('-o', '--only', type=click.STRING, multiple=True, callback=process_skip_only,
                    help="variables to process, skipping all others")
