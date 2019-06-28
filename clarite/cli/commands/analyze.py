import click


@click.group()
def analyze():
    pass


@analyze.command()
def ewas():
    """Run EWAS"""
    print("Ran EWAS")
    pass
