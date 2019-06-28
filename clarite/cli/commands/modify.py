import click


@click.group()
def modify():
    pass


@modify.command()
def recode():
    """Replace some values in the data"""
    print("Replaced some values")
    pass
