import click

from .commands import analyze, describe, modify


@click.group()
def entry_point():
    pass


entry_point.add_command(analyze)
entry_point.add_command(describe)
entry_point.add_command(modify)
