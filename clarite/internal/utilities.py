from typing import Optional, List


def _validate_skip_only(columns, skip: Optional[List[str]] = None, only: Optional[List[str]] = None):
    """Validate use of the 'skip' and 'only' parameters, returning a valid list of columns to filter"""
    if skip is not None and only is not None:
        raise ValueError("It isn't possible to specify 'skip' and 'only' at the same time.")
    elif skip is not None and only is None:
        invalid_cols = set(skip) - set(columns)
        if len(invalid_cols) > 0:
            raise ValueError(f"Invalid columns passed to 'skip': {', '.join(invalid_cols)}")
        columns = [c for c in columns if c not in set(skip)]
    elif skip is None and only is not None:
        invalid_cols = set(only) - set(columns)
        if len(invalid_cols) > 0:
            raise ValueError(f"Invalid columns passed to 'only': {', '.join(invalid_cols)}")
        columns = [c for c in columns if c in set(only)]

    if len(columns) == 0:
        raise ValueError("No columns available for filtering")

    return columns
