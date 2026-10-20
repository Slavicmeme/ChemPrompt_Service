import pytest

from app.services.cancellation import JobCancelled, cancel_scope, check_cancelled


def test_check_is_noop_outside_scope():
    check_cancelled()  # no scope installed -> nothing happens


def test_cancel_scope_raises_after_request():
    flag = {"cancel": False}
    with cancel_scope(lambda: flag["cancel"]):
        check_cancelled()
        flag["cancel"] = True
        with pytest.raises(JobCancelled):
            check_cancelled()
    check_cancelled()  # scope restored on exit


def test_prompt_building_is_a_cancellation_point(smiles):
    from app.services.descriptor_service import DescriptorEngine

    engine = DescriptorEngine.fit(smiles)
    with cancel_scope(lambda: True), pytest.raises(JobCancelled):
        engine.prompts(smiles, engine.full_flag())
