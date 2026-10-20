from chemprompt_core.llm_model import LLMModel
from app.services.descriptor_service import DescriptorEngine


def test_restored_engine_matches_original_filter(smiles):
    original = LLMModel.__new__(LLMModel)
    original._filter_descriptors(smiles)                 # untouched research implementation
    engine = DescriptorEngine.fit(smiles)
    assert engine.descriptor_names == list(original._descriptor_funcs.keys())
    restored = DescriptorEngine.from_names(engine.descriptor_names)
    assert restored.property_names == original.property_names
    flag = [i % 3 == 0 for i in range(len(engine.descriptor_names))]
    for s in smiles[:10]:
        assert restored.prompt(s, flag) == original._create_prompt(s, flag)
        assert restored.prompt(s, engine.full_flag()) == original._create_prompt(s, engine.full_flag())


def test_prompt_format(smiles):
    e = DescriptorEngine.fit(smiles)
    p = e.prompt("CCO", e.full_flag())
    assert p.startswith("SMILES: CCO | ")
    assert e.prompt("CCO", [False] * len(e.descriptor_names)) == "SMILES: CCO"
