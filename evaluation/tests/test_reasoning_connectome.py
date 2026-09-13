"""Routing correctness only: no learned routing or task-level validation."""
from dataclasses import FrozenInstanceError, replace
import json
import pytest

from fifth_layer.reasoning_connectome import DynamicReasoningConnectome, ConnectomeConfig, ConnectomeContext
from fifth_layer.orchestration import ReasonerRegistry, ReasonerOutput, OrchestrationContext
from evaluation.tests.test_reasoner_orchestration import registration
from evaluation.tests.test_common_evidence_state import item, build


def reg(identity='r', family='physics', **kwargs):
    return replace(registration(identity, **kwargs), reasoner_family=family)


def route(state=None, registrations=(), config=None):
    return DynamicReasoningConnectome(config or ConnectomeConfig()).route(
        build() if state is None else state, ReasonerRegistry(registrations))


def test_empty_registry_empty_graph():
    graph = route()
    assert not graph.nodes and not graph.edges and not graph.selected_reasoner_ids
    assert graph.to_json() == route().to_json()


def test_node_references_existing_registration():
    graph = route(registrations=(reg(),))
    node = graph.nodes['r']
    assert node['node_id'] == node['reasoner_id'] == 'r'
    assert node['reasoner_family'] == 'physics' and node['version'] == '0.1'
    assert 'execute' not in node


def test_duplicate_reasoner_rejected():
    with pytest.raises(ValueError, match='duplicate'):
        route(registrations=(reg(), reg()))


def test_duplicate_selected_id_rejected():
    graph = route(registrations=(reg(),))
    with pytest.raises(ValueError, match='duplicate'):
        replace(graph, selected_reasoner_ids=('r', 'r'))


def test_same_input_same_graph_and_scores():
    state = build(item('p', 'physical', 'possible'))
    a, b = route(state, (reg(),)), route(state, (reg(),))
    assert a == b and a.relevance_scores == b.relevance_scores
    assert a.to_json() == b.to_json()
    assert json.loads(a.to_json()) == a.to_dict()


@pytest.mark.parametrize('family,source,status,extra', [
    ('physics', 'physical', 'possible', {}),
    ('temporal', 'temporal', None, {}),
    ('occlusion', 'occlusion', None, {}),
    ('sensory', 'sensory', 'expected', {'modality': 'auditory'}),
    ('learned_representation', 'learned_representation', 'learned_signal', {}),
    ('documentary', 'documentary', 'reported_result', {}),
])
def test_explicit_source_presence_raises_relevance(family, source, status, extra):
    r = reg(family=family)
    baseline = route(registrations=(r,))
    graph = route(build(item('e', source, status, **extra)), (r,))
    assert graph.relevance_scores['r'] > baseline.relevance_scores['r']
    assert graph.relevance_breakdown['r']['evidence_match_refs'] == ('e',)


def test_missing_and_unknown_not_false_or_zero():
    empty = route(registrations=(reg(),))
    assert empty.relevance_scores['r'] == .2
    assert 'missing_context' in empty.uncertainty['routing_context']
    unknown = route(build(item('p', 'physical', 'unknown')), (reg(),))
    assert unknown.relevance_scores['r'] > 0
    assert unknown.context_descriptors[0]['epistemic_status'] == 'unknown'
    assert 'partial_context' in unknown.uncertainty['routing_context']


def test_unavailable_and_empty_feed_do_not_raise_relevance():
    state = build(item('p', 'physical', 'unavailable'), present_families=('physical',))
    assert route(state, (reg(),)).relevance_scores['r'] == .2


def test_non_selection_produces_no_contradiction():
    router, registry, state = DynamicReasoningConnectome(), ReasonerRegistry((reg(),)), build()
    graph = router.route(state, registry)
    assert graph.exclusion_reasons['r']['reason'] == 'below_relevance_threshold'
    assert router.execute(state, registry, graph).results == ()
    assert not state.evidence_items


@pytest.mark.parametrize('source,status,value,extra', [
    ('physical', 'possible', {'finding': 'collision_possible'}, {}),
    ('sensory', 'expected', {'observed': False}, {'modality': 'auditory'}),
    ('learned_representation', 'learned_signal', {'temporal_change_score': .8}, {}),
])
def test_routing_does_not_promote_status_or_confidence(source, status, value, extra):
    evidence = item('e', source, status, value=value, confidence=.1, **extra)
    state = build(evidence)
    before = state.to_json()
    graph = route(state, (reg(required_evidence=(source,)),))
    assert graph.relevance_scores['r'] > .5
    assert state.to_json() == before
    assert state.evidence_items[0] == evidence
    assert state.evidence_items[0].confidence == .1


def test_provenance_refs_and_breakdown_preserved():
    evidence = item('p', 'physical', 'possible')
    state = build(evidence)
    graph = route(state, (reg(),))
    assert graph.source_evidence_refs == state.source_references
    assert graph.source_evidence_refs['p']['provenance'] == evidence.provenance
    breakdown = graph.relevance_breakdown['r']
    assert breakdown['base_relevance'] == .2 and breakdown['evidence_match'] == .4
    assert graph.edges[0]['source_evidence_refs'] == ('p',)


def test_score_bounded_and_clipping_inspectable():
    ctx = OrchestrationContext(build(item('p', 'physical', 'possible')), available_capabilities=('cpu',))
    config = ConnectomeConfig(base_relevance=1., evidence_match_increment=1.,
        category_match_increment=1., capability_match_increment=1.)
    graph = route(ctx, (reg(required_evidence_types=('summary',), required_capabilities=('cpu',)),), config)
    assert graph.relevance_scores['r'] == 1.
    assert graph.relevance_breakdown['r']['unclipped_relevance'] == 4.
    assert graph.relevance_breakdown['r']['clipping_adjustment'] == -3.


@pytest.mark.parametrize('config', [
    {'base_relevance': -1}, {'min_relevance': 1.1}, {'evidence_match_increment': float('nan')},
    {'capability_match_increment': float('inf')}, {'max_selected_reasoners': -1},
    {'max_selected_reasoners': True}, {'family_evidence': {'custom': ('experience',)}},
])
def test_invalid_config_rejected(config):
    with pytest.raises(ValueError):
        ConnectomeConfig(**config)


def test_threshold_inclusive_and_zero_limit():
    config = ConnectomeConfig(base_relevance=.2, min_relevance=.2)
    assert route(registrations=(reg(),), config=config).selected_reasoner_ids == ('r',)
    assert not route(registrations=(reg(),), config=replace(config, max_selected_reasoners=0)).selected_reasoner_ids


def test_limit_and_ties_use_id_not_registration_or_priority():
    config = ConnectomeConfig(min_relevance=0., max_selected_reasoners=2)
    regs = (reg('z', priority=-1), reg('b', priority=0), reg('a', priority=9))
    a, b = route(registrations=regs, config=config), route(registrations=tuple(reversed(regs)), config=config)
    assert a == b and a.selected_reasoner_ids == ('a', 'b')
    assert a.exclusion_reasons['z']['reason'] == 'selection_limit'


@pytest.mark.parametrize('changes,status', [
    ({'required_evidence': ('visual',)}, 'missing_required_evidence'),
    ({'enabled': False}, 'disabled'),
    ({'coordinate_frame_id': 'meters'}, 'incompatible_context'),
    ({'required_capabilities': ('gpu',)}, 'unsupported_input'),
    ({'required_statuses': {'physical': ('observed',)}}, 'missing_required_evidence'),
])
def test_high_relevance_cannot_override_step15(changes, status):
    graph = route(build(item('p', 'physical', 'possible')), (reg(**changes),), ConnectomeConfig(base_relevance=1.))
    assert graph.relevance_scores['r'] == 1.
    assert not graph.selected_reasoner_ids
    assert graph.exclusion_reasons['r']['eligibility']['status'] == status


def test_bridge_reuses_step15_order_and_failure_semantics():
    calls = []
    def fail(c):
        calls.append('a')
        raise RuntimeError('synthetic')
    def succeed(c):
        calls.append('b')
        return ReasonerOutput(structured_output={'confidence': .1, 'finding': 'collision_possible'})
    registry = ReasonerRegistry((reg('b', execute=succeed, priority=-10), reg('a', execute=fail, priority=10)))
    state = build(item('p', 'physical', 'possible'))
    router = DynamicReasoningConnectome()
    graph = router.route(state, registry)
    result = router.execute(state, registry, graph)
    assert calls == ['a', 'b']
    assert [r.execution_status for r in result.results] == ['failed', 'executed']
    assert registry.registrations[0].reasoner_id == 'b'
    assert result.results[1].output.structured_output['confidence'] == .1


def test_same_source_activates_nodes_without_evidence_duplication():
    state = build(item('p', 'physical', 'possible'))
    graph = route(state, (reg('a'), reg('b')))
    assert len(graph.edges) == 2 and len(graph.source_evidence_refs) == 1
    assert len(state.evidence_items) == 1
    assert graph.relevance_scores['a'] == graph.relevance_scores['b']
    assert all(e['relation'] == 'context_activates_reasoner' for e in graph.edges)


def test_duplicate_matching_sources_do_not_accumulate_bonus():
    a = route(build(item('a', 'physical', 'possible')), (reg(),))
    b = route(build((item('a', 'physical', 'possible'), item('b', 'physical', 'possible'))), (reg(),))
    assert a.relevance_scores == b.relevance_scores


def test_caller_mutation_does_not_change_graph():
    matches = {'future': ['physical']}
    router = DynamicReasoningConnectome(ConnectomeConfig(family_evidence=matches))
    graph = router.route(build(item('p', 'physical', 'possible')), ReasonerRegistry((reg(family='future'),)))
    before = graph.to_json()
    matches['future'].append('visual')
    exported = graph.to_dict()
    exported['nodes']['r']['version'] = 'changed'
    assert graph.to_json() == before
    with pytest.raises(TypeError):
        graph.relevance_scores['r'] = 0
    with pytest.raises(FrozenInstanceError):
        graph.timestamp = 3.


@pytest.mark.parametrize('changes', [{'session_id': 'other'}, {'scene_id': 'other'}, {'timestamp': 3.}])
def test_snapshot_context_mismatch_rejected_at_bridge(changes):
    state, registry, router = build(), ReasonerRegistry((reg(),)), DynamicReasoningConnectome()
    graph = router.route(state, registry)
    with pytest.raises(ValueError, match='snapshot'):
        router.execute(state, registry, replace(graph, **changes))


def test_future_source_rejected_before_routing():
    with pytest.raises(ValueError):
        route(build(item(timestamp=3.)), (reg(),))


@pytest.mark.parametrize('payload', [object(), b'raw', {'tensor': [1.]}, {'embedding': list(range(100000))}])
def test_raw_payload_not_stored(payload):
    with pytest.raises(ValueError):
        ctx = OrchestrationContext(build(), configuration={'payload': payload})
        route(ctx, (reg(),))


def test_custom_scientific_family_without_parser():
    state = build((item('r', 'documentary', 'reported_result'), item('c', 'documentary', 'author_claim')))
    graph = route(state, (reg(family='future-claim', optional_evidence=('documentary',)),))
    assert graph.selected_reasoner_ids == ('r',)
    assert {d['epistemic_status'] for d in graph.context_descriptors} == {'reported_result', 'author_claim'}


def test_no_adaptive_state_between_calls():
    router, registry = DynamicReasoningConnectome(), ReasonerRegistry((reg(),))
    empty, physical = build(), build(item('p', 'physical', 'possible'))
    first = router.route(empty, registry)
    active = router.route(physical, registry)
    router.execute(physical, registry, active)
    assert router.route(empty, registry) == first
    assert router.route(physical, registry) == active
    assert first.relevance_scores != active.relevance_scores


def test_experience_payload_does_not_drive_scores():
    first = route(registrations=(reg(optional_evidence=('experience',)),))
    state = build(item('history', 'experience', None, value={'success': True}))
    second = route(state, (reg(optional_evidence=('experience',)),))
    assert first.relevance_scores == second.relevance_scores


def test_forged_selection_cannot_execute_disabled_reasoner():
    state, registry = build(), ReasonerRegistry((reg(enabled=False),))
    router = DynamicReasoningConnectome()
    graph = router.route(state, registry)
    forged = replace(graph, selected_reasoner_ids=('r',), exclusion_reasons={})
    with pytest.raises(ValueError):
        router.execute(state, registry, forged)


def test_registry_metadata_change_rejected():
    router, state = DynamicReasoningConnectome(), build()
    registry = ReasonerRegistry((reg(),))
    graph = router.route(state, registry)
    changed = ReasonerRegistry((replace(reg(), version='0.2'),))
    with pytest.raises(ValueError):
        router.execute(state, changed, graph)
    with pytest.raises(ValueError):
        DynamicReasoningConnectome(ConnectomeConfig(min_relevance=0.)).execute(state, registry, graph)


def test_partial_context_remains_inspectable():
    ctx = ConnectomeContext.from_state(build(item(timestamp=None, provenance={})))
    assert 'partial_context' in ctx.uncertainty['routing_context']
    assert ctx.descriptors[0]['timestamp'] is None


def test_relevance_order_reaches_executor_without_registry_mutation():
    calls = []
    def call(identity):
        def execute(c):
            calls.append(identity)
            return ReasonerOutput()
        return execute
    registry = ReasonerRegistry((reg('a', family='temporal', execute=call('a'), priority=-10),
                                reg('z', execute=call('z'), priority=10)))
    state = build(item('p', 'physical', 'possible'))
    router = DynamicReasoningConnectome(ConnectomeConfig(min_relevance=0.))
    graph = router.route(state, registry)
    router.execute(state, registry, graph)
    assert calls == ['z', 'a']
    assert registry.registrations[0].priority == -10


def test_category_and_capability_edges_are_descriptive():
    state = build(item('paper', 'documentary', 'author_claim', evidence_type='table_evidence'))
    ctx = OrchestrationContext(state, available_capabilities=('offline',))
    graph = route(ctx, (reg(family='future-table', required_evidence_types=('table_evidence',),
                            required_capabilities=('offline',)),))
    assert graph.relevance_breakdown['r']['category_match'] == .2
    assert graph.relevance_breakdown['r']['capability_match'] == .1
    assert {edge['rule'] for edge in graph.edges} == {'category_match', 'capability_match'}


def test_extracted_context_routes_identically():
    state = build(item('p', 'physical', 'possible'))
    assert route(state, (reg(),)) == route(ConnectomeContext.from_state(state), (reg(),))
