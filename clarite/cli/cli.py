import click

from .commands import analyze_cli, describe_cli, io_cli, modify_cli, plot_cli, process_cli


@click.group()
def entry_point():
    pass


entry_point.add_command(analyze_cli)
entry_point.add_command(describe_cli)
entry_point.add_command(io_cli)
entry_point.add_command(modify_cli)
entry_point.add_command(plot_cli)
entry_point.add_command(process_cli)
