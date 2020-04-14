import click

from ...modules import utils


@click.group(name='utils')
def utils_cli():
    pass


@utils_cli.command(help="Install R Packages used by CLARITE")
def install_r_packages():
    utils.setup_r_packages()
