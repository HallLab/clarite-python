import click


@click.group()
def cli():
    pass


@cli.command()
def hello():
    click.echo('Hello')


@cli.command()
def goodbye():
    click.echo('goodbye')
