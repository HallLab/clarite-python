import click

from clarite import __version__

from .commands import analyze_cli, describe_cli, load_cli, modify_cli, plot_cli


@click.group()
@click.version_option(version=__version__)
def entry_point():
    pass


entry_point.add_command(analyze_cli)
entry_point.add_command(describe_cli)
entry_point.add_command(load_cli)
entry_point.add_command(modify_cli)
entry_point.add_command(plot_cli)
