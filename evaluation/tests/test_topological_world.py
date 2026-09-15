"""Synthetic contract and temporal correspondence checks, not task evaluation."""
from dataclasses import FrozenInstanceError, replace
import json

import pytest

from fifth_layer.world_model.complex_systems import ComplexSystemsBuilder, InteractionEdge, CoupledSubsystemCandidate
from fifth_layer.world_model.topological_world import TopologicalWorldBuilder, VERSION
from evaluation.tests.test_complex_systems import relation
from evaluation.tests.test_cross_modal_consequences import latent, snapshot

B = TopologicalWorldBuilder()


def state(t=1., pairs=((0, 1),), status='possible', signal='contact_candidate'):
    p = replace(latent(snapshot(t=t, positions=tuple(range(8)))),
                relations=tuple(relation(f'o{a}', f'o{b}', signal, status) for a, b in pairs))
    return ComplexSystemsBuilder().build(p)


def build(a=((0, 1),), b=((0, 1),)):
    return B.build(state(1., a), state(2., b))


def test_semantic_persistence_despite_source_id_change():
    result = build()
    edge, = result.edges
    assert edge.status == 'persistent'
    assert edge.previous_interaction_ids != edge.current_interaction_ids
    assert edge.object_ids == ('o0', 'o1')
    assert all(n.status == 'persistent' for n in result.nodes)
    assert result.schema_version == VERSION


@pytest.mark.parametrize('a,b,expected', [([], [(0, 1)], 'appeared'), ([(0, 1)], [], 'disappeared')])
def test_representation_gain_loss(a, b, expected):
    result = build(a, b)
    assert result.transition.edge_states[0].status == expected
    assert all(n.status == expected for n in result.transition.node_states)
    assert bool(result.edges) == bool(b)
    assert 'representation_change_not_event_occurrence' in result.uncertainty


@pytest.mark.parametrize('a,b,kind', [
    ([(0, 1)], [(0, 1)], 'persistent_component'),
    ([(0, 1)], [(0, 1), (1, 2)], 'expanded_component'),
    ([(0, 1), (1, 2)], [(0, 1)], 'contracted_component'),
    ([(0, 1), (2, 3)], [(0, 1), (1, 2), (2, 3)], 'merge_candidate'),
    ([(0, 1), (1, 2), (2, 3)], [(0, 1), (2, 3)], 'split_candidate'),
    ([], [(0, 1)], 'appeared_component'),
    ([(0, 1)], [], 'disappeared_component'),
    ([(0, 1)], [(1, 2)], 'reconfigured_component'),
    ([(0, 1), (2, 3)], [(0, 2), (1, 3)], 'reconfigured_component'),
])
def test_component_overlap_rules(a, b, kind):
    result = build(a, b)
    component, = result.transition.component_correspondences
    assert component.correspondence_type == kind
    assert (component.status == 'indeterminate') == (kind == 'reconfigured_component')
    assert set(component.source_edge_ids) <= {e.topology_edge_id for e in result.transition.edge_states}


def test_disjoint_components_are_not_matched():
    result = build([(0, 1)], [(2, 3)])
    assert {c.correspondence_type for c in result.transition.component_correspondences} == {'appeared_component', 'disappeared_component'}


def test_no_nodes_from_isolated_physical_objects():
    assert {n.object_id for n in build().nodes} == {'o0', 'o1'}
    assert not build([], []).nodes


@pytest.mark.parametrize('status', ['indeterminate', 'unavailable', 'unsupported'])
@pytest.mark.parametrize('side', ['previous', 'current', 'both'])
def test_uncertain_status_never_asserts_change(status, side):
    a = state(1., status=status if side in ('previous', 'both') else 'possible')
    b = state(2., status=status if side in ('current', 'both') else 'possible')
    result = B.build(a, b)
    assert result.transition.edge_states[0].status == 'indeterminate'
    assert all(n.status == 'indeterminate' for n in result.transition.node_states)
    assert 'source_relation_indeterminate' in result.uncertainty


def test_multiedges_remain_distinct():
    def multi(t):
        p = replace(latent(snapshot(t=t)), relations=(relation(), relation(signal='support_candidate')))
        return ComplexSystemsBuilder().build(p)
    result = B.build(multi(1.), multi(2.))
    assert len(result.edges) == 2
    assert len({e.interaction_type for e in result.edges}) == 2


def hyper(t):
    s = state(t)
    e = replace(s.interactions[0], object_ids=('o0', 'o1', 'o2'))
    c = replace(s.subsystems[0], object_ids=e.object_ids, interaction_ids=(e.interaction_id,))
    return replace(s, interactions=(e,), subsystems=(c,), stability_assessments=(), transition_candidates=())


def test_hyperedge_not_pairwise_expanded():
    result = B.build(hyper(1.), hyper(2.))
    assert len(result.edges) == 1
    assert len(result.edges[0].object_ids) == 3
    assert len(result.nodes) == 3


def test_ordering_and_repeated_build_determinism():
    a, b = state(1., [(0, 1), (2, 3)]), state(2., [(0, 1), (2, 3)])
    def reverse(s):
        return replace(s, interactions=tuple(reversed(s.interactions)), subsystems=tuple(reversed(s.subsystems)))
    assert B.build(a, b).to_json() == B.build(reverse(a), reverse(b)).to_json() == B.build(a, b).to_json()
    assert state(1., [(0, 1), (2, 3)]).to_json() == state(1., [(2, 3), (0, 1)]).to_json()


def test_immutable_detached_json_and_no_input_mutation():
    a, b = state(1.), state(2.)
    original = a.to_json(), b.to_json()
    result = B.build(a, b)
    assert original == (a.to_json(), b.to_json())
    with pytest.raises(FrozenInstanceError):
        result.timestamp = 3.
    with pytest.raises(TypeError):
        result.provenance['hidden_actor_inference'] = True
    detached = result.to_dict()
    detached['provenance']['hidden_actor_inference'] = True
    assert result.provenance['hidden_actor_inference'] is False
    assert json.loads(result.to_json()) == result.to_dict()


@pytest.mark.parametrize('value', [None, {}, (), 2, 'state'])
@pytest.mark.parametrize('side', [0, 1])
def test_typed_required_inputs(value, side):
    args = [state(1.), state(2.)]
    args[side] = value
    with pytest.raises(ValueError):
        B.build(*args)


@pytest.mark.parametrize('timestamp', [None, 1., 0.])
def test_unknown_equal_reverse_time_rejected(timestamp):
    with pytest.raises(ValueError):
        B.build(state(1.), replace(state(2.), timestamp=timestamp))


@pytest.mark.parametrize('name,value', [('session_id', 'other'), ('coordinate_frame_id', 'other'),
                                     ('schema_version', 'wrong'), ('complex_state_id', 'wrong')])
def test_invalid_source_alignment_and_identity(name, value):
    s = state(2.)
    object.__setattr__(s, name, value)
    with pytest.raises(ValueError):
        B.build(state(1.), s)


@pytest.mark.parametrize('name,value', [('scene_id', 'other'), ('timestamp', 3.), ('session_id', 'other'),
                                     ('coordinate_frame_id', 'other'), ('schema_version', 'wrong'), ('interaction_id', 'wrong')])
def test_child_context_and_identity_rejected(name, value):
    s = state(2.)
    object.__setattr__(s.interactions[0], name, value)
    with pytest.raises(ValueError):
        B.build(state(1.), s)


def test_conflicting_semantic_duplicates_rejected():
    s = state(2.)
    e = replace(s.interactions[0], status='indeterminate')
    s = replace(s, interactions=(*s.interactions, e))
    with pytest.raises(ValueError, match='conflicting'):
        B.build(state(1.), s)


def test_compatible_semantic_duplicates_keep_all_sources():
    s = state(2.)
    e = replace(s.interactions[0], source_field_references=('another.explicit.source',))
    s = replace(s, interactions=(*s.interactions, e))
    result = B.build(state(1.), s)
    assert len(result.edges) == 1
    assert len(result.edges[0].current_interaction_ids) == 2


def test_component_extra_object_rejected():
    s = state(2.)
    c = replace(s.subsystems[0], object_ids=('o0', 'o1', 'invented'))
    s = replace(s, subsystems=(c,), stability_assessments=(), transition_candidates=())
    with pytest.raises(ValueError, match='absent objects'):
        B.build(state(1.), s)


def test_component_disconnected_edges_rejected():
    s = state(2., [(0, 1), (2, 3)])
    c = replace(s.subsystems[0], object_ids=('o0', 'o1', 'o2', 'o3'), interaction_ids=tuple(e.interaction_id for e in s.interactions))
    s = replace(s, subsystems=(c,), stability_assessments=(), transition_candidates=())
    with pytest.raises(ValueError, match='disconnected'):
        B.build(state(1.), s)


def test_uncertainty_union_and_separate_lineage():
    a, b = replace(state(1.), uncertainty=('previous_limit',)), replace(state(2.), uncertainty=('current_limit',))
    result = B.build(a, b)
    assert {'previous_limit', 'current_limit'} <= set(result.uncertainty)
    assert result.provenance['previous_uncertainty'] == ('previous_limit',)
    assert result.provenance['current_uncertainty'] == ('current_limit',)


def test_no_feedback_or_scalar_outputs():
    result = build()
    for key in ('causal_inference', 'hidden_actor_inference', 'branch_selection', 'bayesian_update', 'experience_learning_mutation'):
        assert result.provenance[key] is False
    assert not any(key in result.to_dict() for key in ('score', 'entropy', 'betti_numbers', 'complexity'))


def test_snapshot_scene_labels_may_differ():
    result = build()
    assert result.provenance['previous_scene_id'] != result.provenance['current_scene_id']


@pytest.mark.parametrize('status', ['unsupported', 'unavailable'])
def test_excluded_statuses_are_diagnostics_only(status):
    result = B.build(state(1.), state(2., status=status))
    assert not result.edges and not result.nodes
    assert result.transition.edge_states[0].status == 'indeterminate'


@pytest.mark.parametrize('status', ['indeterminate', 'unsupported', 'unavailable'])
@pytest.mark.parametrize('side', [0, 1])
def test_uncertain_relation_against_empty_is_indeterminate(status, side):
    args = [state(1., []), state(2., [])]
    args[side] = state(float(side + 1), status=status)
    result = B.build(*args)
    assert result.transition.edge_states[0].status == 'indeterminate'


def test_missing_component_edge_reference_rejected():
    s = state(2.)
    object.__setattr__(s.subsystems[0], 'interaction_ids', ('missing',))
    with pytest.raises(ValueError):
        B.build(state(1.), s)


def test_nested_provenance_immutable():
    result = build()
    with pytest.raises(TypeError):
        result.edges[0].provenance['current_sources'][0]['status'] = 'unsupported'


def test_node_identity_is_not_inferred_from_same_classes():
    result = build([(0, 1)], [(2, 3)])
    assert {n.status for n in result.transition.node_states} == {'appeared', 'disappeared'}


def test_one_to_one_equal_membership_does_not_claim_edge_persistence():
    result = build([(0, 1), (1, 2)], [(0, 2), (1, 2)])
    assert result.components[0].correspondence_type == 'persistent_component'
    assert {e.status for e in result.transition.edge_states} == {'persistent', 'appeared', 'disappeared'}
