"""Synthetic deterministic execution and existing output compatibility tests."""
from dataclasses import FrozenInstanceError, replace
import json
import pytest

from fifth_layer.orchestration import (OrchestrationContext, ReasonerRegistration,
    ReasonerRegistry, ReasonerOutput, ReasonerOrchestrator)
from fifth_layer.world_model.evidence import EvidenceBundle
from fifth_layer.world_model.hypothesis import Hypothesis
from evaluation.tests.test_common_evidence_state import item, build


def context(evidence=None, **kwargs):
    return OrchestrationContext(build(item() if evidence is None else evidence), **kwargs)


def registration(identity='r', execute=None, **kwargs):
    return ReasonerRegistration(identity, 'synthetic', '0.1',
        execute if execute is not None else lambda c: ReasonerOutput(), **kwargs)


def run(*registrations, ctx=None):
    return ReasonerOrchestrator(ReasonerRegistry(registrations)).run(ctx or context())


def candidate(c, identity='candidate', family='physical', status='possible', producer='r', **kwargs):
    return item(identity, family, status, provenance={
        'producer_reasoner_id': producer, 'session_id': c.session_id,
        'scene_id': c.scene_id, 'timestamp': c.timestamp,
        'derived_from': tuple(e.evidence_id for e in c.common_evidence_state.evidence_items)}, **kwargs)


def test_empty_registry():
    assert run().results == run().plan == ()
    assert run().to_json() == run().to_json()


def test_single_eligible_executes():
    result = run(registration(required_evidence=('visual',)))
    assert result.results[0].execution_status == 'executed'


def test_stable_order_and_same_plan():
    calls = []
    def reg(identity, priority):
        def execute(c):
            calls.append(identity)
            return ReasonerOutput()
        return registration(identity, execute, priority=priority)
    regs = (reg('z', 0), reg('b', 1), reg('a', 1))
    a, b = run(*regs), run(*reversed(regs))
    assert calls == ['z', 'a', 'b'] * 2
    assert a.plan == b.plan
    assert a.to_json() == b.to_json()


def test_duplicate_registry_id_rejected():
    with pytest.raises(ValueError, match='duplicate'):
        ReasonerRegistry((registration(), registration()))


def test_registration_is_local_and_immutable():
    original = ReasonerRegistry()
    updated = original.register(registration())
    assert original.registrations == () and len(updated.registrations) == 1
    with pytest.raises(FrozenInstanceError):
        updated.registrations = ()


@pytest.mark.parametrize('kwargs,status', [
    ({'required_evidence': ('physical',)}, 'missing_required_evidence'),
    ({'enabled': False}, 'disabled'),
    ({'required_evidence_types': ('absent',)}, 'missing_required_evidence'),
    ({'required_capabilities': ('unsupported',)}, 'unsupported_input'),
    ({'supported_schemas': ('future-version',)}, 'unsupported_input'),
    ({'coordinate_frame_id': 'meters'}, 'incompatible_context'),
])
def test_skips_are_not_negative_evidence(kwargs, status):
    def never(c):
        raise AssertionError('ineligible callable was run')
    result = run(registration(execute=never, **kwargs)).results[0]
    assert result.execution_status == 'skipped'
    assert result.eligibility.status == status
    assert result.output is None and not result.error


def test_empty_present_family_is_not_sufficient_evidence():
    ctx = OrchestrationContext(build(present_families=('visual',)))
    assert run(registration(required_evidence=('visual',)), ctx=ctx).results[0].execution_status == 'skipped'


def test_unavailable_not_sufficient_evidence():
    ctx = context(item(status='unavailable'))
    assert run(registration(required_evidence=('visual',)), ctx=ctx).results[0].execution_status == 'skipped'


def test_expected_sensory_not_observed_for_eligibility():
    expected = item('sound', 'sensory', 'expected', modality='auditory', value={'observed': False})
    result = run(registration(required_statuses={'sensory': ('observed',)}), ctx=context(expected))
    assert result.results[0].eligibility.status == 'missing_required_evidence'


def test_failure_isolated_and_input_unchanged():
    ctx = context()
    before = ctx.common_evidence_state.to_json()
    def bad(c):
        c.common_evidence_state.evidence_items[0].value['value'] = 1
    results = run(registration('a', bad), registration('b'), ctx=ctx).results
    assert [r.execution_status for r in results] == ['failed', 'executed']
    assert ctx.common_evidence_state.to_json() == before
    assert results[0].error['type'] == 'TypeError'
    assert results[0].output is None


def test_identity_and_source_provenance():
    ctx = context()
    result = run(registration(), ctx=ctx).results[0]
    assert result.reasoner_id == 'r'
    assert result.provenance['producer_reasoner_id'] == 'r'
    assert result.provenance['registration']['version'] == '0.1'
    assert result.provenance['source_evidence_ids'] == ('visual1',)
    assert result.provenance['context_reference'] == ctx.execution_id


@pytest.mark.parametrize('family,status,value,extra', [
    ('physical', 'possible', {'finding': 'collision_possible'}, {}),
    ('sensory', 'expected', {'observed': False, 'event': 'expected_sound'}, {'modality': 'auditory'}),
    ('learned_representation', 'learned_signal', {'temporal_change_score': .8}, {}),
    ('documentary', 'reported_result', {'statement': 'synthetic'}, {}),
    ('documentary', 'author_claim', {'statement': 'synthetic'}, {}),
    ('documentary', 'aisthesis_inference', {'statement': 'synthetic'}, {}),
])
def test_candidate_status_preserved_without_state_insertion(family, status, value, extra):
    ctx = context()
    supplied = candidate(ctx, family=family, status=status, value=value, **extra)
    result = run(registration(execute=lambda c: ReasonerOutput(evidence_candidates=(supplied,))), ctx=ctx)
    assert result.results[0].execution_status == 'executed'
    assert result.results[0].output.evidence_candidates == (supplied,)
    assert len(ctx.common_evidence_state.evidence_items) == 1
    assert supplied.epistemic_status == status


def test_caller_mutation_does_not_change_result():
    data = {'nested': {'value': 1}}
    result = run(registration(execute=lambda c: ReasonerOutput(structured_output=data)))
    before = result.to_json()
    data['nested']['value'] = 9
    exported = result.to_dict()
    exported['results'][0]['output']['structured_output']['nested']['value'] = 3
    assert result.to_json() == before
    assert json.loads(before) == result.to_dict()


@pytest.mark.parametrize('configuration', [
    {'session_id': 'other'}, {'scene_id': 'other'}, {'source_timestamp': 3.},
])
def test_context_mismatch_rejected(configuration):
    with pytest.raises(ValueError):
        context(configuration=configuration)


@pytest.mark.parametrize('bad', [{'session_id': 'other'}, {'scene_id': 'other'},
    {'timestamp': 3.}, {'coordinate_frame_id': 'meters'}])
def test_output_mismatch_is_explicit_failure(bad):
    result = run(registration(execute=lambda c: ReasonerOutput(structured_output=bad)))
    assert result.results[0].execution_status == 'failed'


def test_agreement_and_order_do_not_increase_confidence():
    output = ReasonerOutput(structured_output={'finding': 'collision_possible', 'confidence': .2})
    result = run(registration('a', lambda c: output, priority=9), registration('b', lambda c: output, priority=-2))
    assert [r.output.structured_output['confidence'] for r in result.results] == [.2, .2]
    assert 'confidence' not in result.to_dict()


def test_unsupported_return_type_fails():
    assert run(registration(execute=lambda c: object())).results[0].execution_status == 'failed'


@pytest.mark.parametrize('payload', [object(), b'bytes', {'tensor': [1.]}, {'embedding': [1., 2.]}])
def test_raw_state_payload_rejected(payload):
    with pytest.raises(ValueError):
        context(configuration={'payload': payload})
    assert run(registration(execute=lambda c: ReasonerOutput(structured_output={'payload': payload}))).results[0].execution_status == 'failed'


def test_bounded_error_does_not_expose_arbitrary_message():
    def fail(c):
        raise RuntimeError('secret-path-and-token' * 10000)
    result = run(registration(execute=fail))
    assert len(json.dumps(result.to_dict()['results'][0]['error'])) < 160
    assert 'secret-path' not in result.to_json()


def test_custom_family_and_capabilities():
    reg = replace(registration(), reasoner_family='future-mini-lab', required_capabilities=('synthetic',))
    assert run(reg, ctx=context(available_capabilities=('synthetic',))).results[0].execution_status == 'executed'


def test_existing_reasoner_and_expected_consequences_compatibility():
    from fifth_layer.reasoners.physics import PhysicsReasoner
    from fifth_layer.world_state import WorldState
    def adapter(c):
        output = PhysicsReasoner().infer_expected_consequences(
            WorldState(data={'position': (0, 0), 'velocity': (1, 2), 'dt': 1}))
        return ReasonerOutput.from_expected_consequences(output)
    result = run(registration(execute=adapter)).results[0]
    assert result.execution_status == 'executed'
    assert result.output.structured_output['predictions']['expected_next_position'] == (1, 2)
    assert result.output.structured_output['epistemic_role'] == 'expected'


def test_existing_evidence_bundle_and_common_state_unchanged():
    source = item()
    bundle = EvidenceBundle('scene', (source,))
    state = build(bundle)
    before = state.to_json()
    run(registration(), ctx=OrchestrationContext(state))
    assert state.to_json() == before and bundle.items == (source,)


def test_invalid_candidate_provenance_fails():
    result = run(registration(execute=lambda c: ReasonerOutput(evidence_candidates=(item('new'),))))
    assert result.results[0].execution_status == 'failed'


def test_existing_hypothesis_preserved():
    ctx = context()
    h = Hypothesis('h', 'scene', 'continued_motion', 'May move', .2, .2, .1,
        2., 1., 3., evidence_for=('visual1',), provenance={
            'producer_reasoner_id': 'r', 'session_id': 'session', 'scene_id': 'scene',
            'timestamp': 2., 'derived_from': ('visual1',)})
    result = run(registration(execute=lambda c: ReasonerOutput(hypotheses=(h,))), ctx=ctx).results[0]
    assert result.execution_status == 'executed'
    assert result.output.hypotheses == (h,)
    assert result.output.hypotheses[0].posterior_probability == .2


def test_configuration_is_detached():
    data = {'nested': {'limit': 1}}
    ctx = context(configuration=data)
    data['nested']['limit'] = 2
    assert ctx.configuration['nested']['limit'] == 1


def test_orchestrator_registry_cannot_be_replaced():
    orchestrator = ReasonerOrchestrator(ReasonerRegistry())
    with pytest.raises(FrozenInstanceError):
        orchestrator.registry = ReasonerRegistry((registration(),))


def test_context_frame_mismatch_rejected():
    with pytest.raises(ValueError):
        context(configuration={'coordinate_frame_id': 'meters'})


def test_unknown_candidate_source_reference_fails():
    def execute(c):
        output = candidate(c)
        output = replace(output, provenance={**output.provenance, 'derived_from': ('missing',)})
        return ReasonerOutput(evidence_candidates=(output,))
    assert run(registration(execute=execute)).results[0].execution_status == 'failed'


def test_duplicate_candidates_fail_without_stopping_next_reasoner():
    def execute(c):
        output = candidate(c)
        return ReasonerOutput(evidence_candidates=(output, output))
    assert [r.execution_status for r in run(registration('a', execute), registration('b')).results] == ['failed', 'executed']


def test_impossible_hypothesis_target_fails():
    def execute(c):
        p = candidate(c).provenance
        return ReasonerOutput(hypotheses=(Hypothesis('h', c.scene_id, 'motion', 'May move',
            .2, .2, .1, 2., 1., 0., provenance=p),))
    assert run(registration(execute=execute)).results[0].execution_status == 'failed'
