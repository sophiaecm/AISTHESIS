"""Synthetic systems-structure contracts; no empirical complexity claim."""
from dataclasses import FrozenInstanceError, replace
import json
from unittest.mock import patch

import pytest

from fifth_layer.world_model.complex_systems import ComplexSystemsBuilder, InteractionEdge
from fifth_layer.world_model.cross_modal_consequences import CrossModalConsequenceBuilder
from fifth_layer.world_model.multiple_futures import MultipleFuturesBuilder
from fifth_layer.world_model.physics_constraints import PhysicsConstraintBundle, PhysicsTransitionAssessment
from evaluation.tests.test_cross_modal_consequences import latent, snapshot, contact, collision, constraint
from evaluation.tests.test_multiple_futures import moving

B = ComplexSystemsBuilder()


def relation(a='o0', b='o1', signal='contact_candidate', status='possible'):
    return dict(subject_id=a, object_id=b, signal=signal, status=status, value='possible', derived_from=('explicit.source',))


def physical(relations=(), dynamics=(), n=2):
    return replace(latent(snapshot(positions=tuple(range(n)))), relations=relations, dynamics=dynamics)


def support(negative=False, uncertain=False):
    c = constraint('support_stability_possibility', 'support_geometry_possible', 'satisfied')
    records = [c]
    if negative:
        records.append(constraint('inertia_consistency', 'abrupt_change_detected', 'violated', ('o0',), 'negative'))
    if uncertain:
        records.append(constraint('inertia_consistency', 'insufficient_history_or_threshold', 'indeterminate', ('o0',), 'unknown'))
    return physical(), PhysicsConstraintBundle(c.scene_id, c.timestamp, tuple(records))


@pytest.mark.parametrize('value', [None, {}, (), 'physical'])
def test_physical_required(value):
    with pytest.raises(ValueError):
        B.build(value)


@pytest.mark.parametrize('name', ['physics_constraints', 'cross_modal_consequences', 'multiple_futures'])
def test_optional_invalid_type(name):
    with pytest.raises(ValueError):
        B.build(physical(), **{name: object()})


@pytest.mark.parametrize('name', ['physics_constraints', 'cross_modal_consequences', 'multiple_futures'])
def test_optional_independent(name):
    p, c = collision()
    sources = dict(physics_constraints=c, cross_modal_consequences=CrossModalConsequenceBuilder().build(p),
                   multiple_futures=MultipleFuturesBuilder().build(p))
    state = B.build(p, **{name: sources[name]})
    assert state.provenance['layer_availability'][name] == 'available'


def test_all_sources_and_no_mutation():
    p, c = collision()
    cross = CrossModalConsequenceBuilder().build(p, c)
    futures = MultipleFuturesBuilder().build(p, c, cross)
    before = (p.to_json(), cross.to_json(), futures.to_json())
    state = B.build(p, c, cross, futures)
    assert state.interactions and state.transition_candidates
    assert before == (p.to_json(), cross.to_json(), futures.to_json())
    assert any(e.source_consequence_ids for e in state.interactions)
    assert any(e.source_future_ids for e in state.interactions)


@pytest.mark.parametrize('changes', [{'scene_id': 'wrong'}, {'timestamp': 1.}, {'timestamp': 3.}])
def test_constraint_context_mismatch(changes):
    p = physical()
    c = PhysicsConstraintBundle(p.scene_id, p.timestamp)
    with pytest.raises(ValueError, match='mismatch'):
        B.build(p, replace(c, **changes))


@pytest.mark.parametrize('key,value', [('session_id', 'wrong'), ('coordinate_frame_id', 'meters'), ('source_timestamp', 99.)])
def test_nested_context_mismatch(key, value):
    p = physical()
    c = constraint()
    c = replace(c, provenance=dict(c.provenance, **{key: value}))
    with pytest.raises(ValueError):
        B.build(p, PhysicsConstraintBundle(p.scene_id, p.timestamp, (c,)))


def test_source_physical_and_objects_mismatch():
    p = contact()
    cross = CrossModalConsequenceBuilder().build(p)
    with pytest.raises(ValueError):
        B.build(replace(p, latent_state_id='wrong'), cross_modal_consequences=cross)
    candidate = replace(cross.candidates[0], object_ids=('absent', 'o0'))
    with pytest.raises(ValueError):
        B.build(p, cross_modal_consequences=replace(cross, candidates=(candidate,)))


def test_transition_assessment_accepted():
    p, c = collision()
    assessment = PhysicsTransitionAssessment('assessment', p.scene_id, p.timestamp, 'past', 1., c,
        {'session_id': p.session_id, 'coordinate_frame_id': p.coordinate_frame_id})
    assert B.build(p, assessment).interactions


def test_empty_and_coexisting_objects_no_interactions():
    for p in (physical(n=0), physical(n=1), physical(n=4), moving()):
        state = B.build(p)
        assert state.interactions == state.subsystems == state.stability_assessments == ()
        assert set(state.provenance['layer_availability'].values()) == {'unavailable'}


@pytest.mark.parametrize('signal,kind', [('contact_candidate', 'contact_relation_candidate'),
    ('collision_risk_geometry', 'collision_relation_candidate'), ('support_candidate', 'support_relation_candidate')])
def test_explicit_relation_mapping(signal, kind):
    state = B.build(physical((relation(signal=signal),)))
    e = state.interactions[0]
    assert e.interaction_type == kind and e.status == 'possible'
    assert e.object_ids == ('o0', 'o1') and e.source_field_references == ('explicit.source',)
    assert e.provenance['undirected'] is True


@pytest.mark.parametrize('status', ['unavailable', 'unsupported'])
def test_nonusable_edges_no_subsystems(status):
    state = B.build(physical((relation(status=status),)))
    assert state.interactions[0].status == status and state.subsystems == ()


def test_indeterminate_edge_propagates():
    state = B.build(physical((relation(status='indeterminate'),)))
    assert state.subsystems[0].status == 'indeterminate'
    assert 'interaction_indeterminate' in state.subsystems[0].uncertainty
    assert state.stability_assessments[0].status == 'indeterminate'


def test_duplicate_and_permutation_determinism():
    r, s = relation(), relation('o1', 'o2')
    p = physical((r, s), n=3)
    a = B.build(p)
    b = B.build(replace(p, relations=(s, r, r), objects=tuple(reversed(p.objects))))
    assert a.to_json() == b.to_json()
    assert a.complex_state_id == b.complex_state_id
    assert len(a.interactions) == 2


def test_disconnected_components_and_connected_chain():
    state = B.build(physical((relation(), relation('o2', 'o3')), n=4))
    assert {s.object_ids for s in state.subsystems} == {('o0', 'o1'), ('o2', 'o3')}
    chain = B.build(physical((relation(), relation('o1', 'o2')), n=3))
    assert chain.subsystems[0].object_ids == ('o0', 'o1', 'o2')
    assert set(chain.subsystems[0].interaction_ids) == {e.interaction_id for e in chain.interactions}


def test_singleton_discontinuity_does_not_create_coupling():
    d = dict(signal='tracking_or_motion_discontinuity', status='possible', object_ids=('o0',), derived_from=('motion.source',))
    assert B.build(physical(dynamics=(d,))).subsystems == ()
    d['object_ids'] = ('o0', 'o1')
    state = B.build(physical(dynamics=(d,)))
    assert state.interactions[0].interaction_type == 'discontinuity_relation_candidate'
    assert state.stability_assessments[0].status == 'unstable_candidate'


@pytest.mark.parametrize('negative,uncertain,expected', [(False, False, 'stable_candidate'),
    (True, False, 'mixed'), (False, True, 'indeterminate')])
def test_stability_positive_negative_unknown(negative, uncertain, expected):
    assert B.build(*support(negative, uncertain)).stability_assessments[0].status == expected


def test_collision_is_instability_candidate_not_failure():
    p, c = collision()
    a = B.build(p, c).stability_assessments[0]
    assert a.status == 'unstable_candidate' and a.counter_indicators
    assert a.provenance['current_observation'] is False


def test_support_relation_alone_not_proven_stability():
    state = B.build(physical((relation(signal='support_candidate'),)))
    assert state.stability_assessments[0].status == 'indeterminate'


def test_ordinary_motion_and_uncertainty_not_instability():
    p = physical((relation(),), dynamics=({'signal': 'sampled_motion_continuity', 'status': 'estimated',
        'object_ids': ('o0',), 'derived_from': ('motion',)},))
    assert B.build(replace(p, uncertainty=('large_unknown',))).stability_assessments[0].status == 'indeterminate'


def test_partial_positive_coverage_not_stable():
    p, c = support()
    p = physical((relation(), relation('o1', 'o2')), n=3)
    assert B.build(p, c).stability_assessments[0].status == 'indeterminate'


def test_grounded_contact_collision_and_unresolved_transitions():
    p, c = collision()
    f = MultipleFuturesBuilder().build(p, c)
    state = B.build(p, c, multiple_futures=f)
    kinds = {t.pattern_type for t in state.transition_candidates}
    assert kinds == {'contact_transition_candidate', 'collision_transition_candidate', 'unresolved_transition'}
    assert all(t.provenance['occurred'] is False for t in state.transition_candidates)
    assert all(t.status == 'indeterminate' for t in state.transition_candidates if t.pattern_type == 'unresolved_transition')
    assert len(f.branches) == 3


def test_continuity_branch_requires_discontinuity_in_subsystem():
    c = constraint('inertia_consistency', 'abrupt_change_detected', 'violated', ('o0',))
    bundle = PhysicsConstraintBundle(c.scene_id, c.timestamp, (c,))
    p = latent(constraints=bundle)
    p = replace(p, relations=(relation(),))
    f = MultipleFuturesBuilder().build(p, bundle)
    state = B.build(p, bundle, multiple_futures=f)
    assert 'continuity_transition_candidate' in {t.pattern_type for t in state.transition_candidates}


def test_cross_modal_only_enriches_does_not_make_observation_or_edge():
    p = contact()
    cross = CrossModalConsequenceBuilder().build(p)
    state = B.build(p, cross_modal_consequences=cross)
    assert state.interactions[0].source_consequence_ids
    assert all(c.status.value != 'observed' for c in cross.candidates)
    without_relation = replace(p, relations=())
    assert B.build(without_relation, cross_modal_consequences=cross).interactions == ()


def test_source_uncertainty_and_missing_layers():
    p = replace(contact(), uncertainty=('source_unknown',))
    state = B.build(p)
    for record in (state, *state.interactions, *state.subsystems, *state.stability_assessments):
        assert 'source_unknown' in record.uncertainty
    assert state.provenance['layer_availability']['multiple_futures'] == 'unavailable'


def test_deep_immutability_and_json():
    state = B.build(contact())
    assert json.loads(state.to_json()) == state.to_dict()
    for record in (state, *state.interactions, *state.subsystems, *state.stability_assessments):
        with pytest.raises(FrozenInstanceError):
            record.schema_version = 'changed'
        with pytest.raises(TypeError):
            record.provenance['causal_discovery'] = True


def test_conflicting_constraint_ids_rejected():
    p, c = collision()
    changed = replace(c, results=(replace(c.results[0], status='violated'),))
    with pytest.raises(ValueError, match='conflicts'):
        B.build(p, changed)


def test_invalid_direct_edge_and_no_scalar_metrics():
    edge = B.build(contact()).interactions[0]
    with pytest.raises(ValueError):
        replace(edge, object_ids=('o0',))
    with pytest.raises(ValueError):
        replace(edge, status='confirmed')
    state = B.build(contact())
    for record in (state, *state.stability_assessments):
        assert not any(name in record.to_dict() for name in ('confidence', 'probability', 'entropy', 'chaos', 'complexity_score'))
    assert state.provenance['cascade'] == state.provenance['propagation'] == 'deferred'


def test_no_clock_or_feedback_calls():
    p = contact()
    with (patch('time.time', side_effect=AssertionError('clock')),
          patch('fifth_layer.world_model.bayesian_belief_state.BayesianBeliefStateBuilder.update', side_effect=AssertionError('update')),
          patch('fifth_layer.world_model.experience_learning_v02.experience_learning_v02', side_effect=AssertionError('learning'))):
        assert B.build(p).to_json() == B.build(p).to_json()


@pytest.mark.parametrize('value', [False, None])
def test_negative_unknown_relation_value_not_candidate(value):
    state = B.build(physical((dict(relation(), value=value),)))
    assert state.interactions == ()
    assert state.provenance['extraction_audit'][0]['reason'] == 'relation_value_negative_or_unknown'


@pytest.mark.parametrize('layer', ['physical', 'constraints', 'cross', 'futures'])
def test_wrong_schema_rejected(layer):
    p, c = collision()
    cross, f = CrossModalConsequenceBuilder().build(p), MultipleFuturesBuilder().build(p)
    source = {'physical': p, 'constraints': c, 'cross': cross, 'futures': f}[layer]
    object.__setattr__(source, 'schema_version', 'unsupported')
    with pytest.raises(ValueError):
        B.build(p, c, cross, f)


def test_no_future_created_edge_or_branch_count_strength():
    p = contact()
    futures = MultipleFuturesBuilder().build(p)
    full = B.build(p, multiple_futures=futures)
    one = B.build(p, multiple_futures=replace(futures, branches=(futures.branches[0],), branch_relations=()))
    assert len(full.interactions) == len(one.interactions) == 1
    assert full.stability_assessments[0].status == one.stability_assessments[0].status
    assert B.build(replace(p, relations=()), multiple_futures=futures).interactions == ()


@pytest.mark.parametrize('modality,event_family', [('acoustic', 'impact'), ('tactile_force', 'contact_impulse'), ('thermal', 'thermal_candidate')])
def test_expected_modality_provenance_not_measurement(modality, event_family):
    p = contact()
    bundle = CrossModalConsequenceBuilder().build(p)
    c = replace(bundle.candidates[0], modality=modality, event_family=event_family, status='expected')
    state = B.build(p, cross_modal_consequences=replace(bundle, candidates=(c,)))
    preserved = state.interactions[0].provenance['consequence_context'][0]
    assert preserved['status'] == 'expected' and preserved['modality'] == modality
    assert state.provenance['current_observation'] is False


def test_duplicate_singleton_audit_is_deterministic():
    d = dict(signal='tracking_or_motion_discontinuity', status='possible', object_ids=('o0',), derived_from=('motion.source',))
    p = physical(dynamics=(d,))
    assert B.build(p).to_json() == B.build(replace(p, dynamics=(d, d))).to_json()
