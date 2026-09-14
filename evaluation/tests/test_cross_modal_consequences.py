"""Offline engineering tests; no task-level or sensory simulation validation."""
from dataclasses import FrozenInstanceError, replace
import json
import os
import subprocess
import sys

import pytest

from fifth_layer.world_model.cross_modal_consequences import (
    ConsequenceModality, ConsequenceStatus, CrossModalConsequenceBuilder,
    VERSION)
from fifth_layer.world_model.physical_state import (
    PhysicalAttribute as A, PhysicalObjectState as O, PhysicalWorldState as W, PhysicalRelation)
from fifth_layer.world_model.latent_physical_state_builder import LatentPhysicalStateBuilder
from fifth_layer.world_model.physics_constraints import PhysicsConstraintResult as R, PhysicsConstraintBundle as B
from fifth_layer.world_model.physics_constraint_engine import PhysicsConstraintEngine, PhysicsConstraintPolicy
from fifth_layer.world_model.hybrid_world_state import HybridWorldState, LearnedRepresentationSignal


def snapshot(t=2., positions=(5, 15), label='glass'):
    objects = []
    for i, x in enumerate(positions):
        refs = (f'tracker[{i}]',)
        attrs = {'track_id': A(i, 'observed', refs),
                 'class_name': A(label, 'observed', refs),
                 'visibility_state': A('observed', 'observed', refs),
                 'bbox': A((x, 0, x+10, 10), 'observed', refs, units='pixels'),
                 'center': A((x+5, 5), 'observed', refs, units='pixels')}
        objects.append(O(f'o{i}', t, attrs))
    return W(f'scene{t}', t, tuple(objects))


def latent(world=None, constraints=None):
    return LatentPhysicalStateBuilder().build(world or snapshot(), constraints,
        session_id='session', coordinate_frame_id='pixels')


def constraint(family='collision_possibility', finding='collision_possible', status='satisfied',
               ids=('o0', 'o1'), identity='c1'):
    return R(identity, family, 'scene2.0', 2., ids, status, finding, 'Synthetic assessment only',
        derived_from=('bbox.source',), provenance={'session_id': 'session', 'coordinate_frame_id': 'pixels'},
        uncertainty=('upstream_uncertainty',))


def collision():
    r = constraint()
    b = B(r.scene_id, r.timestamp, (r,))
    return latent(constraints=b), b


def build(state=None, constraints=None):
    return CrossModalConsequenceBuilder().build(latent() if state is None else state, constraints)


def candidate():
    return build(collision()[0]).candidates[0]


def contact():
    world = snapshot()
    return latent(replace(world, relations=(PhysicalRelation('o0', 'contact_possible', 'o1',
        A(True, 'possible', ('contact.source',), 'explicit structured contact')),)))


@pytest.mark.parametrize('field,value', [
    ('consequence_id', ''), ('scene_id', ' '), ('session_id', ''),
    ('source_physical_state_id', ''), ('rule_id', ''), ('event_family', ''),
    ('description', ''), ('modality', 'visual'), ('status', 'observed'), ('status', 'occurred'),
    ('confidence', -.1), ('confidence', 0.), ('confidence', .5), ('confidence', 1.1),
    ('confidence', float('nan')), ('timestamp', -1), ('timestamp', True),
    ('source_field_references', ()), ('object_ids', ()), ('source_constraint_ids', 'c1'),
    ('provenance', {}), ('provenance', {'embedding': [1, 2]}),
])
def test_invalid_candidate_contract(field, value):
    with pytest.raises(ValueError):
        replace(candidate(), **{field: value})


@pytest.mark.parametrize('status', list(ConsequenceStatus))
def test_status_contract_never_observation(status):
    c = replace(candidate(), status=status)
    assert c.status != 'observed'
    assert c.confidence is None
    assert 'observed' not in c.to_dict()


@pytest.mark.parametrize('modality', list(ConsequenceModality))
def test_modality_vocabulary(modality):
    assert replace(candidate(), modality=modality).to_dict()['modality'] == modality.value


def test_nested_immutability_and_detachment():
    metadata = {'nested': {'values': ['source']}}
    c = replace(candidate(), provenance=metadata)
    metadata['nested']['values'].append('later')
    assert c.provenance['nested']['values'] == ('source',)
    with pytest.raises(FrozenInstanceError):
        c.status = 'observed'
    with pytest.raises(TypeError):
        c.provenance['nested']['values'] = ()
    b = build(collision()[0])
    with pytest.raises(FrozenInstanceError):
        b.candidates = ()
    with pytest.raises(TypeError):
        b.provenance['truth_decision'] = 'performed'


@pytest.mark.parametrize('metadata', [{'x': 'a'*16385}, {'x': float('inf')}, {'x': object()}])
def test_bounded_metadata(metadata):
    with pytest.raises(ValueError):
        replace(candidate(), provenance=metadata)


def test_metadata_depth_bound():
    metadata = {'x': 0}
    for _ in range(30):
        metadata = {'x': metadata}
    with pytest.raises(ValueError):
        replace(candidate(), provenance=metadata)


@pytest.mark.parametrize('field,value', [('scene_id', 'other'), ('timestamp', 3.),
    ('timestamp', None), ('session_id', 'other'), ('coordinate_frame_id', 'other')])
def test_bundle_context_mismatch(field, value):
    b = build(collision()[0])
    with pytest.raises(ValueError):
        replace(b, candidates=(replace(b.candidates[0], **{field: value}),))


def test_bundle_rejects_duplicate_ids_and_semantics():
    b = build(collision()[0])
    c = b.candidates[0]
    for duplicate in (c, replace(c, consequence_id='different')):
        with pytest.raises(ValueError):
            replace(b, candidates=(c, duplicate))


def test_invalid_bundle_elements():
    with pytest.raises(ValueError):
        replace(build(), candidates=('invalid',))


def test_deterministic_serialization_and_ids():
    state, constraints = collision()
    a, b = build(state, constraints), build(state, constraints)
    assert a.to_json() == b.to_json()
    assert json.loads(a.to_json()) == a.to_dict()
    assert a.candidates[0].to_json() == b.candidates[0].to_json()
    assert replace(a, candidates=tuple(reversed(a.candidates))).to_json() == a.to_json()
    assert len({c.consequence_id for c in a.candidates}) == 3


def test_context_changes_identity():
    state = contact()
    baseline = build(state).candidates[0].consequence_id
    for changes in ({'session_id': 'other'}, {'coordinate_frame_id': 'other'},
                    {'timestamp': 3.}, {'scene_id': 'other'}, {'latent_state_id': 'other'}):
        assert build(replace(state, **changes)).candidates[0].consequence_id != baseline


def test_contact_does_not_invent_impact_or_heat():
    cs = build(contact()).candidates
    assert [(c.modality, c.event_family, c.status) for c in cs] == [
        ('tactile_force', 'sustained_contact_force', 'possible')]
    assert 'contact_duration_and_load_unknown' in cs[0].uncertainty


def test_collision_candidates_are_separate_possibilities():
    cs = build(collision()[0]).candidates
    assert {(c.modality, c.event_family) for c in cs} == {
        ('acoustic', 'impact'), ('tactile_force', 'contact_impulse'), ('kinesthetic', 'abrupt_motion_change')}
    assert all(c.status == 'possible' and c.confidence is None for c in cs)
    assert all(c.object_ids == ('o0', 'o1') for c in cs)


@pytest.mark.parametrize('family,finding', [('implausible_displacement', 'kinematic_outlier'),
                                         ('inertia_consistency', 'abrupt_change_detected')])
def test_motion_discontinuity_remains_ambiguous(family, finding):
    r = constraint(family, finding, 'violated', ('o0',))
    cs = build(latent(constraints=B(r.scene_id, r.timestamp, (r,)))).candidates
    assert {c.event_family for c in cs} == {'acceleration_change', 'inertial_force_change'}
    assert all(c.status == 'indeterminate' for c in cs)
    assert all('tracking_or_camera_change_may_explain_signal' in c.uncertainty for c in cs)


@pytest.mark.parametrize('status', ['indeterminate', 'unsupported', 'not_applicable', 'violated'])
def test_bare_constraint_status_does_not_generate_event(status):
    r = constraint(status=status)
    b = B(r.scene_id, r.timestamp, (r,))
    assert build(latent(constraints=b), b).candidates == ()


@pytest.mark.parametrize('status', ['unsupported', 'not_applicable'])
def test_optional_constraint_blocks_stale_summary(status):
    state, b = collision()
    r = replace(b.results[0], status=status)
    assert build(state, replace(b, results=(r,))).candidates == ()


def test_optional_indeterminate_constraint_downgrades_summary():
    state, b = collision()
    r = replace(b.results[0], status='indeterminate', finding='collision_indeterminate')
    cs = build(state, replace(b, results=(r,))).candidates
    assert cs and all(c.status == 'indeterminate' for c in cs)
    assert all('source_constraint_conflict_or_ambiguity' in c.uncertainty for c in cs)


def test_source_provenance_and_uncertainty():
    state, b = collision()
    for c in build(state, b).candidates:
        assert c.source_physical_state_id == state.latent_state_id
        assert c.source_constraint_ids == ('c1',)
        assert 'bbox.source' in c.source_field_references
        assert c.rule_id == 'conditional_collision_risk_geometry'
        assert c.rule_version == c.schema_version == VERSION
        assert c.coordinate_frame_id == 'pixels' and c.session_id == 'session'
        assert c.provenance['source_schema_version'] == state.schema_version
        assert c.provenance['rule_id'] == c.rule_id
        assert c.provenance['source_constraints']['c1'][1]['provenance'] == b.results[0].provenance
        assert 'upstream_uncertainty' in c.uncertainty
        assert set(state.uncertainty) <= set(c.uncertainty)


def test_relation_field_provenance():
    c = build(contact()).candidates[0]
    assert 'contact.source' in c.source_field_references
    assert c.source_constraint_ids == ()
    assert any('.relations[' in ref for ref in c.source_field_references)


@pytest.mark.parametrize('changes', [{'status': 'unknown'}, {'value': None}, {'value': False}, {'value': 1},
                                    {'derived_from': ()}, {'signal': 'shattering'}])
def test_missing_or_unsupported_prerequisite_skipped(changes):
    state = contact()
    source = dict(state.relations[0], **changes)
    assert build(replace(state, relations=(source,))).candidates == ()


def test_untraceable_constraint_signal_skipped():
    state, _ = collision()
    assert build(replace(state, active_constraints=())).candidates == ()


@pytest.mark.parametrize('signal', ['overlaps', 'support_candidate', 'friction', 'deformation'])
def test_no_unimplemented_family_inference(signal):
    state = contact()
    assert build(replace(state, relations=(dict(state.relations[0], signal=signal),))).candidates == ()


def test_semantic_objects_and_missing_modalities_generate_no_facts():
    state = latent(snapshot(label='glass'))
    before = state.to_json()
    result = build(state)
    assert result.candidates == ()
    assert state.to_json() == before
    assert result.provenance['missing_prerequisite_policy'] == 'skip_not_negative_evidence'
    assert result.provenance['truth_decision'] == 'not_performed'
    assert 'sensor_availability_not_evaluated' in result.uncertainty
    assert build(latent(snapshot(positions=()))).candidates == ()


def test_no_observation_hidden_actor_or_belief_claims():
    b = build(collision()[0])
    for c in b.candidates:
        assert set(c.object_ids) <= {'o0', 'o1'}
        for text in ('sound detected', 'sound was heard', 'force measured', 'temperature observed',
                     'vibration sensed', 'hidden actor', 'hidden person'):
            assert text not in c.description
        assert 'supports' not in c.to_dict() and 'contradicts' not in c.to_dict()
        assert c.provenance['truth_decision'] == 'not_performed'


def test_learned_signal_does_not_change_consequences():
    state, b = collision()
    signal = LearnedRepresentationSignal('synthetic', 2., 'session', 'scene2.0', 'rep', 16,
        {'semantic_claim': 'impact heard and hidden person'}, temporal_change_score=100.)
    assert build(HybridWorldState(state, b, signal)).to_json() == build(state, b).to_json()
    assert build(HybridWorldState(latent(), learned_signal=signal)).candidates == ()


@pytest.mark.parametrize('changes', [{'scene_id': 'other'}, {'timestamp': 3.},
                                    {'session_id': 'other'}, {'coordinate_frame_id': 'other'}])
def test_hybrid_mismatch_protection(changes):
    signal = LearnedRepresentationSignal('synthetic', 2., 'session', 'scene2.0', 'rep', 16, {'source': 'fixture'})
    with pytest.raises(ValueError):
        build(HybridWorldState(latent(), learned_signal=replace(signal, **changes)))


@pytest.mark.parametrize('changes', [{'scene_id': 'other'}, {'timestamp': 3.}, {'timestamp': None}])
def test_constraint_context_rejected(changes):
    with pytest.raises(ValueError):
        build(latent(), B(**({'scene_id': 'scene2.0', 'timestamp': 2.} | changes)))


@pytest.mark.parametrize('metadata', [{'session_id': 'other', 'coordinate_frame_id': 'pixels'},
    {'session_id': 'session', 'coordinate_frame_id': 'other'},
    {'session_id': 'session', 'coordinate_frame_id': 'pixels', 'source_timestamp': 3.}])
def test_constraint_provenance_alignment(metadata):
    state, b = collision()
    with pytest.raises(ValueError):
        build(state, replace(b, results=(replace(b.results[0], provenance=metadata),)))


def test_future_physical_source_rejected():
    with pytest.raises(ValueError):
        build(replace(latent(), provenance={'source_timestamp': 3.}))


def test_unknown_timestamp_retained():
    b = build(replace(contact(), timestamp=None))
    assert b.timestamp is None and b.candidates[0].timestamp is None
    assert 'temporal_alignment_unknown' in b.candidates[0].uncertainty


def test_constraint_object_mismatch_rejected():
    state, b = collision()
    with pytest.raises(ValueError):
        build(state, replace(b, results=(replace(b.results[0], object_ids=('o0', 'unknown')),)))


def test_unknown_dynamics_object_never_created():
    state = latent()
    d = {'signal': 'tracking_or_motion_discontinuity', 'status': 'possible',
         'object_ids': ('hidden',), 'derived_from': ('c1',)}
    assert build(replace(state, dynamics=(d,))).candidates == ()


def test_multiple_sources_order_and_duplicate_suppression():
    state = contact()
    first = state.relations[0]
    second = dict(first, derived_from=('second.source',))
    a = build(replace(state, relations=(first, second, first)))
    b = build(replace(state, relations=(second, first)))
    assert len(a.candidates) == 2
    assert a.to_json() == b.to_json()


def test_real_engine_collision_and_retrospective_assessment():
    previous, current = snapshot(1., (0, 30)), snapshot()
    engine = PhysicsConstraintEngine()
    old = latent(previous)
    assessment = engine.assess(current, history=(previous,), session_id='session', coordinate_frame_id='pixels')
    state = LatentPhysicalStateBuilder().build(current, assessment, history=(old,),
        session_id='session', coordinate_frame_id='pixels')
    before = assessment.to_json(), state.to_json()
    cs = build(state, assessment).candidates
    assert len(cs) == 3
    assert all(c.status == 'possible' for c in cs)
    assert all(c.provenance['assessment_id'] == assessment.assessment_id for c in cs)
    assert all(c.provenance['assessment_kind'] == assessment.assessment_kind for c in cs)
    assert before == (assessment.to_json(), state.to_json())
    with pytest.raises(ValueError):
        build(state, replace(assessment, assessment_kind='future_prediction'))


def test_real_engine_motion_change_is_not_force_measurement():
    old, mid, current = snapshot(0., (0,)), snapshot(1., (1,)), snapshot(2., (20,))
    assessment = PhysicsConstraintEngine(PhysicsConstraintPolicy(
        max_velocity_change_pixels_per_second=2.)).assess(current, history=(old, mid),
            session_id='session', coordinate_frame_id='pixels')
    state = latent(current, assessment.constraints)
    cs = build(state, assessment).candidates
    assert len(cs) == 2
    assert all(c.status == 'indeterminate' and c.confidence is None for c in cs)


def test_ambiguous_api_rejected():
    with pytest.raises(ValueError):
        build(HybridWorldState(latent()), B('scene2.0', 2.))
    with pytest.raises(ValueError):
        build('not a physical state')


def test_process_determinism_and_no_model_imports():
    script = '''
import sys
from evaluation.tests.test_cross_modal_consequences import build, collision
print(build(collision()[0]).to_json())
assert not {'torch', 'transformers', 'ultralytics', 'cv2'} & set(sys.modules)
'''
    outputs = [subprocess.run([sys.executable, '-c', script], check=True, capture_output=True,
        text=True, env={**os.environ, 'PYTHONHASHSEED': seed}).stdout for seed in ('1', '42')]
    assert outputs[0] == outputs[1]


def test_optional_source_uncertainty_survives_skipped_generation():
    r = constraint(status='indeterminate')
    b = build(latent(), B(r.scene_id, r.timestamp, (r,)))
    assert b.candidates == () and 'upstream_uncertainty' in b.uncertainty


def test_conflicting_duplicate_source_rejected():
    state = contact()
    with pytest.raises(ValueError):
        build(replace(state, relations=(state.relations[0], dict(state.relations[0], status='indeterminate'))))
