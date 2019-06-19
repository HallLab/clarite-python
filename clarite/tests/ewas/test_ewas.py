from pathlib import Path

import pytest
import pandas as pd

import clarite

DATA_PATH = Path(__file__).parent / 'data'


def test_load_csv(capfd, request):
    # Load without specifying correct separator
    with pytest.raises(ValueError, match="Index SEQN invalid"):
        clarite.io.load_data(DATA_PATH / "PBCD_H.txt", index_col='SEQN')

    # Actually load the data
    df = clarite.io.load_data(DATA_PATH / "PBCD_H.txt", index_col='SEQN', sep="\t")
    out, err = capfd.readouterr()
    assert out == "Loaded 5,932 observations of 18 variables\n"
    assert err == ""

    # Save data for other tests
    request.config.cache.set('data', df.to_json())


def test_ewas(request):
    df = pd.read_json(request.config.cache.get('data', None))
    assert len(df) == 5932

    # Run EWAS
