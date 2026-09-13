"""Synthetic contract tests, not evidence usefulness or scientific validation."""
from dataclasses import FrozenInstanceError, replace
import json

import pytest

from fifth_layer.world_model.evidence import EvidenceItem, EvidenceBundle
from fifth_layer.world_model.common_evidence_state import CommonEvidenceState
from fifth_layer.world_model.common_evidence_state_builder import CommonEvidenceStateBuilder
from fifth_layer.world_model.hybrid_world_state import HybridWorldState
from fifth_layer.world_model.physics_constraints import PhysicsConstraintBundle
from evaluation.tests.test_hybrid_world_state import sources


def item(identity='visual1', family='visual', status='observed', **changes):
    values = dict(evidence_id=identity, scene_id='scene', source_type=family,
        source_component='synthetic', evidence_type='summary', value={'value': None},
        timestamp=2., epistemic_status=status,
        provenance={'session_id': 'session', 'coordinate_frame_id': 'pixels', 'source_id': identity})
    values.update(changes)
    return EvidenceItem(**values)


def build(evidence=(), **changes):
    args = dict(scene_id='scene', session_id='session', timestamp=2., coordinate_frame_id='pixels', evidence=evidence)
    args.update(changes)
    return CommonEvidenceStateBuilder().build(**args)


def test_multiple_families(sources):
    sensory = item('sound', 'sensory', 'expected', modality='auditory', value={'observed': False})
    state = build((item(), sensory), hybrid_state=HybridWorldState(*sources))
    assert {e.source_type.value for e in state.evidence_items} == {
        'visual', 'sensory', 'physical', 'physics_constraint', 'learned_representation'}
    assert all(isinstance(e, EvidenceItem) for e in state.evidence_items)


def test_deterministic_order_and_serialization():
    a, b = item('a'), item('b')
    first, second = build((a, b)), build((b, a))
    assert first == second
    assert first.to_json() == second.to_json()
    assert json.loads(first.to_json()) == first.to_dict()


@pytest.mark.parametrize('status', ['observed', 'estimated', 'possible', 'unknown', 'unavailable'])
def test_visual_status_preserved(status):
    assert build(item(status=status)).evidence_items[0].epistemic_status == status


def test_physics_possibility_remains_possibility(sources):
    state = build(physics_constraints=sources[1])
    evidence = state.evidence_items[0]
    assert evidence.evidence_id == 'c1'
    assert evidence.epistemic_status == 'assessment'
    assert evidence.value['finding'] == 'collision_possible'
    assert evidence.value['status'] == 'satisfied'
    assert evidence.value['uncertainty'] == ('actual_contact_unknown',)


def test_learned_and_physical_are_separate(sources):
    state = build(hybrid_state=HybridWorldState(*sources))
    by_family = {e.source_type.value: e for e in state.evidence_items}
    assert by_family['learned_representation'].epistemic_status == 'learned_signal'
    assert by_family['physical'].epistemic_status == 'mixed'
    assert by_family['physical'].value['objects'][0]['attributes']['center']['value'] is None
    assert by_family['learned_representation'].value['source_model'] == 'vjepa2'
    assert state.provenance['hybrid']['source_references']['explicit_physical'] == sources[0].latent_state_id


def test_expected_sensory_is_not_observed():
    expected = item('sound', 'sensory', 'expected', modality='auditory', value={'observed': False})
    assert build(expected).evidence_items[0] == expected
    assert build(expected).source_references['sound']['epistemic_status'] == 'expected'


def test_unavailable_sensory_not_contradiction():
    unavailable = item('sound', 'sensory', 'unavailable', modality='auditory', value={'observation_available': False})
    state = build(unavailable)
    assert state.availability['sensory'] == 'unavailable'
    assert state.evidence_items[0].contradicts == ()


def test_missing_family_is_unavailable():
    state = build(item())
    assert state.availability['documentary'] == 'unavailable'
    assert state.availability['learned_representation'] == 'unavailable'
    assert all(e.source_type.value != 'documentary' for e in state.evidence_items)


def test_unknown_and_zero_distinct():
    a = item('unknown', status='unknown', value={'value': None})
    b = item('zero', value={'value': 0})
    values = {e.evidence_id: e.value['value'] for e in build((a, b)).evidence_items}
    assert values == {'unknown': None, 'zero': 0}


def test_provenance_and_uncertainty_preserved():
    source = item(value={'uncertainty': ('depth_unknown',)}, provenance={'session_id': 'session', 'source': {'page': 3}})
    bundle = EvidenceBundle('scene', (source,), {'producer': 'collection'})
    state = build(bundle)
    assert state.evidence_items[0] == source
    assert state.source_references['visual1']['provenance'] == source.provenance
    assert state.uncertainty['sources']['visual1']['item.value.uncertainty'] == ('depth_unknown',)
    assert {'producer': 'collection'} in [dict(v) for v in state.provenance.values()]


def test_same_statement_different_sources_not_collapsed():
    a = item('a', value={'statement': 'may be occluded'})
    b = item('b', 'documentary', 'author_claim', value={'statement': 'may be occluded'})
    assert len(build((a, b)).evidence_items) == 2


@pytest.mark.parametrize('conflict', [False, True])
def test_duplicate_id_rejected(conflict):
    a = item()
    b = replace(a, value={'different': True}) if conflict else a
    with pytest.raises(ValueError, match='duplicate'):
        build((a, b))


@pytest.mark.parametrize('changes', [
    {'scene_id': 'other'}, {'timestamp': 3.},
    {'provenance': {'session_id': 'other'}},
    {'provenance': {'coordinate_frame_id': 'meters'}},
    {'provenance': {'source': {'session_id': 'other'}}},
    {'provenance': {'source_timestamp': 3.}},
    {'timestamp': 1., 'provenance': {'source_timestamp': 2.}},
])
def test_alignment_rejects(changes):
    with pytest.raises(ValueError):
        build(item(**changes))


def test_missing_context_is_explicit():
    state = build(item(timestamp=None, provenance={}))
    assert set(state.uncertainty['integration']) >= {
        'source_context_incomplete', 'coordinate_frame_unspecified', 'temporal_alignment_unknown'}


def test_documentary_without_frame():
    source = item('paper', 'documentary', 'reported_result', provenance={'session_id': 'session', 'doi': 'synthetic'})
    state = build(source, coordinate_frame_id=None)
    assert state.evidence_items[0] == source
    assert 'coordinate_frame_unspecified' not in state.uncertainty['integration']


@pytest.mark.parametrize('status', ['reported_result', 'author_claim', 'aisthesis_inference', 'unknown',
    'limitation', 'method', 'measurement', 'dataset_reference', 'figure_evidence', 'table_evidence', 'equation_relation'])
def test_scientific_categories_preserved(status):
    source = item(status, 'documentary', status, evidence_type='custom_mini_lab_category')
    assert build(source).evidence_items[0] == source


def test_report_and_claim_remain_distinct():
    state = build((item('r', 'documentary', 'reported_result'), item('c', 'documentary', 'author_claim')))
    assert {e.epistemic_status for e in state.evidence_items} == {'reported_result', 'author_claim'}


def test_immutable_detached_state():
    payload = {'value': {'measurement': None}}
    metadata = {'session_id': 'session', 'page': 1}
    source = item(value=payload, provenance=metadata)
    bundle = EvidenceBundle('scene', (source,))
    state = build(bundle)
    before = state.to_json()
    payload['value']['measurement'] = 0
    metadata['page'] = 9
    exported = state.to_dict()
    exported['evidence']['items'][0]['value']['value']['measurement'] = 9
    assert state.to_json() == before
    assert state.evidence_items[0] == source and bundle.items == (source,)
    with pytest.raises(FrozenInstanceError):
        state.timestamp = 7.
    with pytest.raises(TypeError):
        state.evidence_items[0].value['value']['measurement'] = 1


@pytest.mark.parametrize('payload', [object(), b'bytes', {'embedding': [1., 2.]},
    {'array': [1.]}, {'text': 'x'*16385}, {'values': list(range(50001))}])
def test_raw_or_oversized_payload_rejected(payload):
    with pytest.raises(ValueError):
        build(item(value={'data': payload}))


def test_empty_present_distinct():
    assert build().availability['visual'] == 'unavailable'
    assert build(present_families=('visual',)).availability['visual'] == 'available'
    assert build(physics_constraints=PhysicsConstraintBundle('scene', 2.)).availability['physics_constraint'] == 'available'


def test_hybrid_inputs_unchanged(sources):
    hybrid = HybridWorldState(*sources)
    before = hybrid.to_json()
    build(hybrid_state=hybrid)
    assert hybrid.to_json() == before


def test_existing_legacy_evidence_preserved():
    legacy = EvidenceItem('legacy', 'scene', 'motion', 'provider', 'motion_state',
                          {'motion_state': 'moving'}, 2., supports=('continued_motion',))
    state = build(legacy)
    assert state.evidence_items[0] == legacy
    assert state.evidence_items[0].epistemic_status is None
    assert state.source_references['legacy']['source_family'] == 'motion'


def test_no_contradiction_resolution():
    a, b = item('a', supports=('claim',)), item('b', contradicts=('claim',))
    assert build((a, b)).evidence_items == (a, b)


def test_direct_constructor_validates():
    bundle = EvidenceBundle('scene', (item(provenance={'session_id': 'wrong'}),))
    with pytest.raises(ValueError):
        CommonEvidenceState('scene', 'session', 2., bundle)


def test_learned_promotion_rejected():
    with pytest.raises(ValueError):
        item('learned', 'learned_representation', 'observed')


def test_hybrid_components_cannot_be_overridden(sources):
    with pytest.raises(ValueError):
        build(hybrid_state=HybridWorldState(*sources), learned_signal=sources[2])


def test_prediction_target_is_not_source_leakage():
    source = item(value={'target_timestamp': 3., 'observed': False}, status='possible')
    assert build(source).evidence_items[0] == source


@pytest.mark.parametrize('changes', [{'session_id': 'other'}, {'scene_id': 'other'},
    {'coordinate_frame_id': 'meters'}, {'timestamp': 1.}])
def test_hybrid_context_mismatch(sources, changes):
    with pytest.raises(ValueError):
        build(hybrid_state=HybridWorldState(*sources), **changes)


def test_bundle_frame_mismatch():
    bundle = EvidenceBundle('scene', (item(),), {'coordinate_frame_id': 'meters'})
    with pytest.raises(ValueError):
        build(bundle)


def test_direct_learned_stale_time_rejected():
    with pytest.raises(ValueError):
        build(item('learned', 'learned_representation', 'learned_signal', timestamp=1.))


def test_assessment_provenance_preserved(sources):
    from fifth_layer.world_model.physical_state import PhysicalWorldState
    from fifth_layer.world_model.physics_constraint_engine import PhysicsConstraintEngine
    assessment = PhysicsConstraintEngine().assess(PhysicalWorldState('scene', 2.),
        session_id='session', coordinate_frame_id='pixels')
    state = build(physics_constraints=assessment)
    assert state.provenance['physics_constraints']['assessment_id'] == assessment.assessment_id
    assert state.provenance['physics_constraints']['provenance'] == assessment.provenance


def test_bundle_provenance_session_mismatch():
    with pytest.raises(ValueError):
        build(EvidenceBundle('scene', (), {'session_id': 'other'}))
