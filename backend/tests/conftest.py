import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["REPRESENTATION_MODE"] = "mock"  # tests never call the Aya service, whatever the runtime env says
os.environ["GA_MODE"] = "mock"
os.environ["OPENAI_API_KEY"] = ""  # tests never call the real OpenAI API

SMILES = [
    "CCO", "c1ccccc1O", "CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "CCCCCCCCCC(=O)O",
    "O=C(O)CC(O)(CC(=O)O)C(=O)O", "c1ccc2ccccc2c1", "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "ClC(Cl)(Cl)Cl", "NCCc1ccc(O)c(O)c1",
    "C1CCCCC1", "Oc1ccccc1Cl", "c1ccncc1", "c1ccc2[nH]ccc2c1", "CC(C)O", "CCOC(=O)C", "c1ccc(cc1)C(=O)O",
    "CC1=CC=CC=C1", "OCC(O)CO", "CCN(CC)CC", "C1CCOC1", "c1ccsc1", "CC(=O)N", "CCCCO", "c1ccc(cc1)N",
    "COc1ccccc1", "c1cc(ccc1O)O", "CC(C)(C)O", "C1CCNCC1", "c1ccc(cc1)Cl", "OC(=O)c1ccccc1O", "CCCC(=O)O",
    "c1ccc2c(c1)cccc2O", "CC(=O)c1ccccc1", "C(CO)N", "c1cnc2ccccc2c1", "CC#N", "CS(=O)C", "c1ccoc1", "CCCCCC",
]


@pytest.fixture
def smiles():
    return list(SMILES)


@pytest.fixture
def y_reg(smiles):
    rng = np.random.RandomState(0)
    return rng.randn(len(smiles), 1)

