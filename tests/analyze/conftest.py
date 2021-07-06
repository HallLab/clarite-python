from pathlib import Path

from pandas_genomics import scalars, sim

import clarite
import pytest

TESTS_PATH = Path(__file__).parent.parent
DATA_PATH = TESTS_PATH / "test_data_files"


# Dataset fixtures
@pytest.fixture
def data_fpc():
    # Load the data
    df = clarite.load.from_csv(DATA_PATH / "fpc_data.csv", index_col=None)
    # Process data
    df = clarite.modify.make_continuous(df, only=["x", "y"])
    return df


@pytest.fixture()
def data_NHANES():
    df = clarite.load.from_csv(DATA_PATH / "nhanes_data.csv", index_col=None)
    df = clarite.modify.make_binary(df, only=["HI_CHOL", "RIAGENDR"])
    df = clarite.modify.make_categorical(df, only=["race", "agecat"])
    return df


@pytest.fixture()
def data_NHANES_withNA():
    df = clarite.load.from_csv(DATA_PATH / "nhanes_NAs_data.csv", index_col=None)
    df = clarite.modify.make_binary(df, only=["HI_CHOL", "RIAGENDR"])
    df = clarite.modify.make_categorical(df, only=["race", "agecat"])
    return df


@pytest.fixture()
def data_NHANES_lonely():
    df = clarite.load.from_csv(DATA_PATH / "nhanes_lonely_data.csv", index_col=None)
    df = clarite.modify.make_binary(df, only=["HI_CHOL", "RIAGENDR"])
    df = clarite.modify.make_categorical(df, only=["race", "agecat"])
    return df


@pytest.fixture()
def genotype_case_control_add_add_main():
    var1 = scalars.Variant(chromosome="1", position=123456, id="rs1", ref="T", alt="A")
    var2 = scalars.Variant(chromosome="10", position=30123, id="rs2", ref="A", alt="C")
    model = sim.BAMS.from_model(
        eff1=sim.SNPEffectEncodings.ADDITIVE,
        eff2=sim.SNPEffectEncodings.ADDITIVE,
        snp1=var1,
        snp2=var2,
    )
    genotypes = model.generate_case_control()
    # Add random gt
    for i in range(2, 50):
        var = scalars.Variant(ref="A", alt="C")
        genotypes[f"var{i}"] = sim.generate_random_gt(
            var, alt_allele_freq=[0.01 * i], n=2000
        )
    return genotypes


@pytest.fixture()
def genotype_case_control_rec_rec_onlyinteraction():
    var1 = scalars.Variant(chromosome="1", position=123456, id="rs1", ref="T", alt="A")
    var2 = scalars.Variant(chromosome="10", position=30123, id="rs2", ref="A", alt="C")
    model = sim.BAMS.from_model(
        eff1=sim.SNPEffectEncodings.RECESSIVE,
        eff2=sim.SNPEffectEncodings.RECESSIVE,
        snp1=var1,
        snp2=var2,
        main1=0,
        main2=0,
        interaction=1.0,
    )
    genotypes = model.generate_case_control(n_cases=5000, n_controls=2000, snr=0.01)
    # Add random gt
    for i in range(2, 50):
        var = scalars.Variant(ref="A", alt="C")
        genotypes[f"var{i}"] = sim.generate_random_gt(
            var, alt_allele_freq=[0.01 * i], n=7000
        )
    return genotypes
