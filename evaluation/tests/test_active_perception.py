"""Synthetic advisory contract checks; no measured active-perception benefit."""
from dataclasses import FrozenInstanceError, replace
import json
import socket
from unittest.mock import patch

import pytest

from fifth_layer.world_model.active_perception import ActivePerceptionCore, ObservationCue, ObservationTargetCandidate, ObservationRequestPlan, VERSION, POLICY
from fifth_layer.world_model.complex_systems import ComplexSystemsBuilder, TransitionPatternCandidate
from fifth_layer.world_model.topological_world import TopologicalWorldBuilder
from evaluation.tests.test_topological_world import state
from evaluation.tests.test_cross_modal_consequences import latent, snapshot
from evaluation.tests.test_complex_systems import relation

B = ActivePerceptionCore()


def inputs(a=((0, 1),), b=((0, 1),), status='possible'):
    previous, current = state(1., a), state(2., b, status=status)
    return TopologicalWorldBuilder().build(previous, current), current


def supported(objects_uncertainty=False, dynamics=(), relations=()):
    p = latent(snapshot(t=2.))
    objects = tuple(dict(o, uncertainty=('tracking_unresolved',) if objects_uncertainty else ()) for o in p.objects)
    p = replace(p, objects=objects, relations=relations, dynamics=dynamics)
    cs = ComplexSystemsBuilder().build(p)
    return TopologicalWorldBuilder().build(state(1., []), cs), cs, p


def test_determinate_persistent_topology_has_empty_plan():
    t, _ = inputs()
    result = B.build(t)
    assert not result.cues and not result.targets
    assert 'empty_plan_not_certainty' in result.uncertainty
    assert result.source_physical_state_id is None


def test_indeterminate_relation_medium_without_false_independence():
    t, _ = inputs(status='indeterminate')
    result = B.build(t)
    target, = [x for x in result.targets if x.target_scope == 'relation']
    assert target.priority_band == 'medium'
    assert len(target.cue_ids) == 2
    assert {c.cue_type for c in result.cues} >= {'indeterminate_relation', 'relation_change_ambiguity'}


@pytest.mark.parametrize('a,b', [([], [(0, 1)]), ([(0, 1)], [])])
def test_change_is_low_and_not_event(a, b):
    t, _ = inputs(a, b)
    result = B.build(t)
    assert len(result.targets) == 1
    assert result.targets[0].priority_band == 'low'
    assert result.cues[0].cue_type == 'relation_change_ambiguity'
    assert 'representation_change_not_event_occurrence' in result.cues[0].uncertainty


@pytest.mark.parametrize('a,b,kind', [
    ([(0, 1)], [(1, 2)], 'reconfigured_component'),
    ([(0, 1), (1, 2), (2, 3)], [(0, 1), (2, 3)], 'split_candidate'),
    ([(0, 1), (2, 3)], [(0, 1), (1, 2), (2, 3)], 'merge_candidate'),
])
def test_component_cues(a, b, kind):
    t, _ = inputs(a, b)
    result = B.build(t)
    cue, = [c for c in result.cues if c.cue_type == 'component_reconfiguration']
    assert cue.provenance['correspondence_type'] == kind
    assert next(x for x in result.targets if cue.cue_id in x.cue_ids).priority_band == 'medium'


def test_direct_component_edge_corroboration_high():
    t, _ = inputs(status='indeterminate')
    result = B.build(t)
    target, = [x for x in result.targets if x.target_scope == 'subsystem']
    assert target.priority_band == 'high'
    assert result.targets[0] == target


def test_complex_interaction_and_subsystem_cues_deduplicate_targets():
    t, cs = inputs(status='indeterminate')
    result = B.build(t, cs)
    relation_target, = [x for x in result.targets if x.target_scope == 'relation']
    assert len(relation_target.cue_ids) == 3
    assert relation_target.priority_band == 'medium'
    assert any(c.cue_type == 'subsystem_uncertainty' for c in result.cues)
    assert any(c.provenance['source_layer'] == 'complex_systems' and c.cue_type == 'indeterminate_relation' for c in result.cues)


def test_stability_and_transition_grounded_through_subsystem():
    _, cs = inputs()
    assessment = replace(cs.stability_assessments[0], status='indeterminate')
    transition = TransitionPatternCandidate(cs.subsystems[0].subsystem_id, 'unresolved_transition', 'indeterminate',
                                            ('explicit.source',), {}, ('unresolved',), {})
    cs = replace(cs, stability_assessments=(assessment,), transition_candidates=(transition,))
    t = TopologicalWorldBuilder().build(state(1.), cs)
    result = B.build(t, cs)
    assert {'stability_uncertainty', 'transition_uncertainty'} <= {c.cue_type for c in result.cues}
    target, = result.targets
    assert target.target_scope == 'subsystem' and target.priority_band == 'medium'
    assert target.object_ids == cs.subsystems[0].object_ids


def test_physical_object_uncertainty_explicit_only():
    t, cs, p = supported(objects_uncertainty=True)
    result = B.build(t, cs, p)
    assert len(result.targets) == 2
    assert all(x.target_scope == 'object' and x.priority_band == 'low' for x in result.targets)
    assert {o for x in result.targets for o in x.object_ids} == {'o0', 'o1'}


@pytest.mark.parametrize('status', ['observed', 'estimated', 'possible', 'indeterminate', 'unknown'])
def test_explicit_discontinuity(status):
    d = dict(signal='tracking_or_motion_discontinuity', status=status, object_ids=('o0',), derived_from=('constraint.source',))
    result = B.build(*supported(dynamics=(d,)))
    assert len(result.targets) == 1
    assert result.cues[0].provenance['discontinuity'] is True


def test_physical_relation_uncertainty():
    r = dict(relation(), uncertainty=('relation_unresolved',))
    result = B.build(*supported(relations=(r,)))
    assert any(c.cue_type == 'physical_state_uncertainty' and c.object_ids == ('o0', 'o1') for c in result.cues)


@pytest.mark.parametrize('signal', ['generic_motion', 'newly_observed_object', 'proximity', 'interesting', 'hidden_actor'])
def test_no_signal_or_class_only_cues(signal):
    d = dict(signal=signal, status='observed', object_ids=('o0',), derived_from=('explicit.source',))
    result = B.build(*supported(dynamics=(d,)))
    assert not result.targets


def test_no_optional_inputs_required_for_topological_cue():
    t, _ = inputs([], [(0, 1)])
    assert B.build(t).targets


def test_input_mutation_detached_serialization_and_immutability():
    args = supported(objects_uncertainty=True)
    before = [a.to_json() for a in args]
    result = B.build(*args)
    assert before == [a.to_json() for a in args]
    for record in (result, *result.cues, *result.targets):
        assert record.schema_version == VERSION
        with pytest.raises(FrozenInstanceError):
            record.schema_version = 'bad'
        with pytest.raises(TypeError):
            record.provenance['test'] = True
    output = result.to_dict()
    output['provenance']['advisory_only'] = False
    assert result.provenance['advisory_only'] is True
    assert json.loads(result.to_json()) == result.to_dict()


def test_ordering_determinism():
    t, cs, p = supported(objects_uncertainty=True)
    result = B.build(t, cs, p)
    p = replace(p, objects=tuple(reversed(p.objects)))
    cs = replace(cs, interactions=tuple(reversed(cs.interactions)), subsystems=tuple(reversed(cs.subsystems)))
    t = replace(t, nodes=tuple(reversed(t.nodes)), edges=tuple(reversed(t.edges)))
    assert result.to_json() == B.build(t, cs, p).to_json()
    assert [c.cue_id for c in result.cues] == sorted(c.cue_id for c in result.cues)


@pytest.mark.parametrize('value', [None, {}, [], 'topology', 1])
def test_bad_primary_type(value):
    with pytest.raises(ValueError):
        B.build(value)


@pytest.mark.parametrize('name,value', [('schema_version', 'wrong'), ('topological_state_id', 'wrong'),
    ('timestamp', 99.), ('session_id', 'other'), ('coordinate_frame_id', 'other'), ('source_complex_state_id', 'wrong')])
def test_corrupt_primary_rejected(name, value):
    t, _ = inputs()
    object.__setattr__(t, name, value)
    with pytest.raises(ValueError):
        B.build(t)


@pytest.mark.parametrize('name,value', [('scene_id', 'other'), ('timestamp', 3.), ('session_id', 'other'),
    ('coordinate_frame_id', 'other'), ('source_complex_state_id', 'wrong')])
def test_reconstructed_topology_alignment_rejected(name, value):
    t, _ = inputs()
    t = replace(t, **{name: value})
    with pytest.raises(ValueError):
        B.build(t)


@pytest.mark.parametrize('value', [{}, 'complex', 2])
def test_wrong_complex_type(value):
    t, _ = inputs()
    with pytest.raises(ValueError):
        B.build(t, value)


@pytest.mark.parametrize('name,value', [('scene_id', 'other'), ('timestamp', 3.), ('session_id', 'other'),
    ('coordinate_frame_id', 'other'), ('schema_version', 'wrong'), ('complex_state_id', 'wrong')])
def test_complex_mismatch(name, value):
    t, cs = inputs()
    object.__setattr__(cs, name, value)
    with pytest.raises(ValueError):
        B.build(t, cs)


@pytest.mark.parametrize('name,value', [('scene_id', 'other'), ('timestamp', 3.), ('session_id', 'other'),
    ('coordinate_frame_id', 'other'), ('schema_version', 'wrong'), ('latent_state_id', 'wrong')])
def test_physical_mismatch(name, value):
    t, cs, p = supported()
    p = replace(p, **{name: value})
    with pytest.raises(ValueError):
        B.build(t, cs, p)


def test_physical_requires_complex_lineage():
    t, _, p = supported()
    with pytest.raises(ValueError):
        B.build(t, physical_state=p)


def test_malformed_dynamic_reference():
    t, cs, p = supported()
    p = replace(p, dynamics=(dict(signal='tracking_or_motion_discontinuity', status='possible', object_ids=('invented',)),))
    with pytest.raises(ValueError):
        B.build(t, cs, p)


def test_malformed_topological_object_reference():
    t, _ = inputs()
    e = replace(t.transition.edge_states[0], object_ids=('invented', 'o0'))
    tr = replace(t.transition, edge_states=(e,))
    t = replace(t, transition=tr, edges=(e,))
    with pytest.raises(ValueError):
        B.build(t)


@pytest.mark.parametrize('name', ['target_scope', 'priority_band'])
def test_target_vocabulary_validation(name):
    target = B.build(inputs([], [(0, 1)])[0]).targets[0]
    with pytest.raises(ValueError):
        replace(target, **{name: 'invented'})


def test_plan_rejects_unknown_cue_reference():
    result = B.build(inputs([], [(0, 1)])[0])
    target = replace(result.targets[0], cue_ids=('unknown',))
    with pytest.raises(ValueError):
        replace(result, targets=(target,))


def test_symbolic_only_no_geometry_leaks():
    result = B.build(*supported(objects_uncertainty=True))
    forbidden = {'bbox', 'box', 'mask', 'center', 'coordinates', 'pan', 'tilt', 'zoom', 'trajectory', 'velocity', 'probability', 'score', 'risk'}
    def check(value):
        if isinstance(value, dict):
            assert not forbidden.intersection(value)
            for v in value.values(): check(v)
        elif isinstance(value, list):
            for v in value: check(v)
    check(result.to_dict())


def test_no_side_effects_policy_and_network():
    t, cs = inputs(status='indeterminate')
    with patch.object(socket, 'socket', side_effect=AssertionError('network forbidden')):
        result = B.build(t, cs)
    assert all(result.provenance[k] == v for k, v in POLICY.items())
    assert all(x.priority_band in ('high', 'medium', 'low', 'indeterminate') for x in result.targets)


def test_duplicate_physical_cues_and_uncertainty_strings_deduplicate():
    d = dict(signal='tracking_or_motion_discontinuity', status='possible', object_ids=('o0',), uncertainty=('u', 'u'))
    result = B.build(*supported(dynamics=(d, d)))
    assert len(result.cues) == len(result.targets) == 1


def test_uncertainty_and_provenance_preserved():
    t, cs, p = supported(objects_uncertainty=True)
    result = B.build(t, cs, p)
    assert set(t.uncertainty) <= set(result.uncertainty)
    for c in result.cues:
        assert c.provenance['source_object_ids'] == c.object_ids
        assert c.provenance['source_record_ids'] == c.source_ids
        assert set(c.provenance['source_uncertainty']) <= set(c.uncertainty)
    assert all(set(x.cue_ids) <= {c.cue_id for c in result.cues} for x in result.targets)


@pytest.mark.parametrize('collection', ['nodes', 'edges', 'components'])
def test_incomplete_current_projection_rejected(collection):
    t, _ = inputs()
    with pytest.raises(ValueError):
        B.build(replace(t, **{collection: ()}))


def test_reconstructed_edge_lineage_corruption_rejected_without_optional_source():
    t, _ = inputs()
    e = replace(t.edges[0], current_interaction_ids=('invented_source',))
    tr = replace(t.transition, edge_states=(e,))
    t = replace(t, edges=(e,), transition=tr)
    with pytest.raises(ValueError):
        B.build(t)


def test_same_physical_id_cannot_hide_source_content_mismatch():
    t, cs, p = supported()
    p = replace(p, uncertainty=('invented_uncertainty',))
    with pytest.raises(ValueError, match='source snapshot'):
        B.build(t, cs, p)


def test_multiedge_targets_do_not_collapse_relation_types():
    p = replace(latent(snapshot(t=2.)), relations=(relation(), relation(signal='support_candidate')))
    cs = ComplexSystemsBuilder().build(p)
    t = TopologicalWorldBuilder().build(state(1., []), cs)
    result = B.build(t)
    assert len(result.targets) == 2
    assert len({x.relation_types for x in result.targets}) == 2


def test_hyperedge_target_remains_one_explicit_set():
    from evaluation.tests.test_topological_world import hyper
    t = TopologicalWorldBuilder().build(state(1., []), hyper(2.))
    result = B.build(t)
    assert len(result.targets) == 1
    assert result.targets[0].object_ids == ('o0', 'o1', 'o2')


def test_unrelated_indeterminate_edge_does_not_raise_component_priority():
    previous = state(1., [(0, 1), (4, 5)])
    p = replace(latent(snapshot(t=2., positions=tuple(range(6)))),
                relations=(relation('o1', 'o2'), relation('o4', 'o5', status='indeterminate')))
    current = ComplexSystemsBuilder().build(p)
    # The reconfigured component itself has no indeterminate edge.
    t = TopologicalWorldBuilder().build(previous, current)
    result = B.build(t)
    target, = [x for x in result.targets if x.target_scope == 'subsystem' and x.object_ids == ('o0', 'o1', 'o2')]
    assert target.priority_band == 'medium'
    assert any(x.priority_band == 'high' and x.object_ids == ('o4', 'o5') for x in result.targets)
