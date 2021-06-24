import pandas as pd
import pytest
from statsmodels import datasets

from clarite import describe


@pytest.fixture
def plantTraits():
    data = datasets.get_rdataset("plantTraits", "cluster", cache=True).data
    data.index.name = "ID"
    return data


def test_correlations(plantTraits, capfd):
    # TODO
    describe.correlations(plantTraits, threshold=0.9)
    return


def test_freq_table(plantTraits, capfd):
    # TODO
    return


def test_percent_na(plantTraits):
    result = describe.percent_na(plantTraits)
    correct_result = pd.DataFrame(
        [
            ["pdias", 26.470588],
            ["longindex", 18.382353],
            ["durflow", 0],
            ["height", 0],
            ["begflow", 0],
        ],
        columns=["Variable", "percent_na"],
    )
    pd.testing.assert_frame_equal(result.head(), correct_result)


@pytest.mark.parametrize("dropna", [True, False])
def test_skewness(plantTraits, dropna):
    result = describe.skewness(plantTraits, dropna)
    if dropna:
        correct_result = pd.DataFrame(
            [
                ["pdias", "continuous", 5.156840, 9.610278, 7.235317e-22],
                ["longindex", "continuous", 0.163848, 0.742076, 4.580411e-01],
                ["durflow", "continuous", 2.754286, 8.183515, 2.756827e-16],
                ["height", "continuous", 0.583514, 2.735605, 6.226567e-03],
                ["begflow", "continuous", -0.316648, -1.549449, 1.212738e-01],
            ],
            columns=["Variable", "type", "skew", "zscore", "pvalue"],
        )
    else:
        correct_result = pd.DataFrame(
            [
                ["pdias", "continuous", None, None, None],
                ["longindex", "continuous", None, None, None],
                ["durflow", "continuous", 2.754286, 8.183515, 2.756827e-16],
                ["height", "continuous", 0.583514, 2.735605, 6.226567e-03],
                ["begflow", "continuous", -0.316648, -1.549449, 1.212738e-01],
            ],
            columns=["Variable", "type", "skew", "zscore", "pvalue"],
        )
    pd.testing.assert_frame_equal(result.head(), correct_result)
