import click


@click.group()
def describe():
    pass


@describe.command()
def unique():
    """Get unique values"""
    print("Got unique")
    pass
