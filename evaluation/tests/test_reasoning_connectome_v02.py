"""Offline routing and execution contracts, not task-level routing validation."""
from dataclasses import FrozenInstanceError, replace
import json
from unittest.mock import patch

import pytest

from fifth_layer.reasoning_connectome_v02 import (
    MultiLayerReasoningContext as Context, ReasonerRoutingProfile as Profile,
    ConnectomeConfigV02 as Config, DynamicReasoningConnectomeV02 as Router, LAYERS,
)
from fifth_layer.orchestration import OrchestrationContext, ReasonerRegistry, ReasonerOutput
from fifth_layer.world_model.evidence import EvidenceBundle
from fifth_layer.world_model.common_evidence_state import CommonEvidenceState
from fifth_layer.world_model.physics_constraints import PhysicsConstraintBundle, PhysicsTransitionAssessment
from fifth_layer.world_model.cross_modal_consequences import CrossModalConsequenceBuilder
from fifth_layer.world_model.multiple_futures import MultipleFuturesBuilder
from fifth_layer.world_model.bayesian_belief_state import BayesianBeliefStateBuilder
from fifth_layer.world_model.experience_learning_v02 import experience_learning_v02
from evaluation.tests.test_common_evidence_state import build, item
from evaluation.tests.test_reasoner_orchestration import registration
from evaluation.tests.test_multiple_futures import moving
from evaluation.tests.test_cross_modal_consequences import collision, contact
from evaluation.tests.test_experience_learning_v02 import current, trajectory, historical


def context(evidence=(), **layers):
    source = next((v for v in layers.values() if v is not None), None)
    common = build(evidence, **({} if source is None else dict(scene_id=source.scene_id,
        timestamp=source.timestamp, session_id=getattr(source, 'session_id', 'session'),
        coordinate_frame_id=getattr(source, 'coordinate_frame_id', 'pixels'))))
    return Context(OrchestrationContext(common), **layers)


def full():
    physical, constraints = collision()
    cross = CrossModalConsequenceBuilder().build(physical, constraints)
    futures = MultipleFuturesBuilder().build(physical, constraints, cross)
    belief = BayesianBeliefStateBuilder().initialize(futures, initialization_mode='uniform_uninformative')
    view = experience_learning_v02(belief)
    return context(physical_state=physical, physics_constraints=constraints,
        cross_modal_consequences=cross, multiple_futures=futures, belief_state=belief, experience_view=view)


def experience(enabled=True, cap=.05):
    belief = current()
    return context(belief_state=belief,
        experience_view=experience_learning_v02(belief, trajectory(), enabled=enabled, max_absolute_contribution=cap))


def route(ctx=None, profiles=(), registrations=None, config=None):
    return Router(config or Config()).route(ctx or context(),
        ReasonerRegistry((registration(),) if registrations is None else registrations), profiles)


def cats(ctx, layer):
    return {d['category'] for d in ctx.descriptors if d['layer'] == layer}


@pytest.mark.parametrize('value', [None, {}, build(), 'context'])
def test_orchestration_context_required(value):
    with pytest.raises(ValueError):
        Context(value)


def test_complete_context_and_determinism():
    ctx = full()
    assert ctx.to_json() == full().to_json()
    assert all(v == 'available' for k, v in ctx.layer_availability.items() if k != 'experience')
    assert ctx.layer_availability['experience'] == 'disabled'
    assert json.loads(ctx.to_json()) == ctx.to_dict()


@pytest.mark.parametrize('name', ['physical_state', 'physics_constraints', 'cross_modal_consequences',
                                'multiple_futures', 'belief_state', 'experience_view'])
def test_optional_layer_alone(name):
    ctx = context(**{name: getattr(full(), name)})
    assert getattr(ctx, name) is not None
    assert route(ctx).relevance_scores['r'] == .2


def test_transition_assessment_accepted():
    _, constraints = collision()
    assessment = PhysicsTransitionAssessment('assessment', constraints.scene_id, constraints.timestamp,
        'earlier', 1., constraints, {'session_id': 'session', 'coordinate_frame_id': 'pixels'})
    ctx = context(physics_constraints=assessment)
    assert 'physics_constraints.status.satisfied' in cats(ctx, 'physics_constraints')


@pytest.mark.parametrize('name,value', [('scene_id', 'other'), ('session_id', 'other'),
    ('timestamp', 3.), ('timestamp', 1.), ('coordinate_frame_id', 'meters')])
def test_current_alignment_rejected(name, value):
    p = moving()
    ctx = context(physical_state=p)
    with pytest.raises(ValueError, match='mismatch'):
        replace(ctx, physical_state=replace(p, **{name: value}))


@pytest.mark.parametrize('name', ['physical_state', 'physics_constraints', 'cross_modal_consequences',
                                'multiple_futures', 'belief_state', 'experience_view'])
def test_invalid_optional_type(name):
    with pytest.raises(ValueError):
        Context(OrchestrationContext(build()), **{name: object()})


def test_nested_future_provenance_rejected():
    p = moving()
    with pytest.raises(ValueError, match='future'):
        context(physical_state=replace(p, provenance={'source_timestamp': 99.}))


def test_absence_unavailable_and_no_penalty():
    ctx = context()
    assert all(ctx.layer_availability[l] == 'unavailable' for l in LAYERS if l != 'evidence')
    graph = route(ctx, (Profile('r', preferred_layers=LAYERS),))
    assert graph.relevance_scores['r'] == .2
    assert graph.relevance_breakdown['r']['uncertainty_penalty'] == 0
    assert 'physical.unavailable' in ctx.uncertainty['flags']


@pytest.mark.parametrize('profiles', [(Profile('unknown'),), (Profile('r'), Profile('r'))])
def test_profiles_registered_unique(profiles):
    with pytest.raises(ValueError):
        route(profiles=profiles)


def test_profile_immutable_canonical():
    p = Profile('r', preferred_layers=['belief', 'physical', 'belief'], provenance={'source': ['caller']})
    assert p.preferred_layers == ('belief', 'physical')
    assert p.to_json() == Profile('r', ('physical', 'belief'), provenance={'source': ['caller']}).to_json()
    with pytest.raises(FrozenInstanceError):
        p.routing_role = 'changed'
    with pytest.raises(TypeError):
        p.provenance['source'] = ()
    with pytest.raises(ValueError):
        Profile('r', ('guessed_layer',))


def test_no_profile_base_only_no_name_guessing():
    regs = (registration('physics'), registration('belief'), registration('anything'))
    g = route(full(), registrations=regs)
    assert set(g.relevance_scores.values()) == {.2}
    assert not g.selected_reasoner_ids and not g.edges


def test_evidence_descriptors_exact_status_no_magnitude():
    ctx = context(item('e', 'sensory', 'expected', modality='auditory', confidence=.9))
    for d in ctx.descriptors:
        if d['layer'] == 'evidence':
            assert d['details']['epistemic_status'] == 'expected'
            assert 'confidence' not in d['details'] and 'value' not in d['details']
    g = route(ctx, (Profile('r', preferred_descriptor_categories=('evidence.family.sensory',)),))
    assert g.relevance_scores['r'] == pytest.approx(.3)


def test_physical_descriptors_explicit_signals():
    ctx = context(physical_state=moving())
    assert 'physical.motion_state_present' in cats(ctx, 'physical')
    assert 'physical.contact_candidate_present' not in cats(ctx, 'physical')
    assert 'physical.contact_candidate_present' in cats(context(physical_state=contact()), 'physical')


def test_violation_is_status_not_event():
    _, bundle = collision()
    bundle = replace(bundle, results=(replace(bundle.results[0], status='violated'),))
    ctx = context(physics_constraints=bundle)
    assert 'physics_constraints.status.violated' in cats(ctx, 'physics_constraints')
    assert all('occurred' not in d['details'] for d in ctx.descriptors)


def test_cross_modal_candidates_and_future_relations_preserved():
    ctx = full()
    assert any(c.startswith('cross_modal.modality.') for c in cats(ctx, 'cross_modal'))
    assert 'multiple_futures.relation.mutually_exclusive' in cats(ctx, 'multiple_futures')
    assert all(d['details'].get('epistemic_role') != 'observed' for d in ctx.descriptors)
    graph = route(ctx, (Profile('r', ('cross_modal', 'multiple_futures')),))
    assert graph.provenance['branch_selected'] is False


def test_once_per_layer_and_descriptor_bonus():
    ctx = full()
    p = Profile('r', LAYERS, tuple(d['category'] for d in ctx.descriptors))
    g = route(ctx, (p,))
    parts = g.relevance_breakdown['r']
    assert parts['descriptor_bonus'] == .1
    assert parts['layer_bonuses']['cross_modal'] == .1
    assert parts['layer_bonuses']['experience'] == 0
    assert sum(e['routing_rule'] == 'descriptor_category_presence' for e in g.edges) == 1


def test_evidence_count_and_confidence_do_not_scale():
    p = (Profile('r', ('evidence',), ('evidence.family.visual',)),)
    a = route(context(item()), p)
    b = route(context(tuple(item(str(i), confidence=.99) for i in range(10))), p)
    assert a.relevance_scores == b.relevance_scores


def test_branch_count_does_not_scale():
    f = full().multiple_futures
    one = replace(f, branches=(f.branches[0],), branch_relations=())
    profiles = (Profile('r', ('multiple_futures',)),)
    assert route(context(multiple_futures=one), profiles).relevance_scores == route(context(multiple_futures=f), profiles).relevance_scores


def test_candidate_count_does_not_scale():
    c = full().cross_modal_consequences
    one = replace(c, candidates=(c.candidates[0],))
    profiles = (Profile('r', ('cross_modal',)),)
    assert route(context(cross_modal_consequences=c), profiles).relevance_scores == route(context(cross_modal_consequences=one), profiles).relevance_scores


def test_score_clip_not_normalized_and_tie_order():
    cfg = Config(base_relevance=1., physical_layer_increment=1., max_selected_reasoners=1)
    g = route(context(physical_state=moving()), (Profile('a', ('physical',)), Profile('b', ('physical',))),
        (registration('b'), registration('a')), cfg)
    assert g.relevance_scores == {'a': 1., 'b': 1.}
    assert g.selected_reasoner_ids == ('a',)
    assert g.exclusion_reasons['b']['reason'] == 'selection_limit'
    assert g.relevance_breakdown['a']['clipping_adjustment'] == -1


@pytest.mark.parametrize('settings', [{'min_relevance': -.1}, {'base_relevance': True},
    {'experience_layer_increment': 1.1}, {'descriptor_match_increment': float('nan')},
    {'max_selected_reasoners': -1}, {'max_selected_reasoners': True}])
def test_config_bounds(settings):
    with pytest.raises(ValueError):
        Config(**settings)


@pytest.mark.parametrize('settings,status', [({'enabled': False}, 'disabled'),
    ({'required_capabilities': ('missing',)}, 'unsupported_input'),
    ({'coordinate_frame_id': 'meters'}, 'incompatible_context'),
    ({'required_evidence': ('physical',)}, 'missing_required_evidence'),
    ({'required_evidence': ('sensory',), 'required_statuses': {'sensory': ('observed',)}}, 'missing_required_evidence'),
    ({'required_evidence_types': ('current_observation',)}, 'missing_required_evidence')])
def test_step15_ineligible_never_overridden(settings, status):
    g = route(full(), (Profile('r', LAYERS),), (registration(**settings),), Config(base_relevance=1.))
    assert not g.selected_reasoner_ids
    assert g.exclusion_reasons['r']['reason'] == 'step15_ineligible'
    assert g.exclusion_reasons['r']['eligibility']['status'] == status


def test_experience_cannot_supply_required_current_evidence():
    g = route(experience(), (Profile('r', ('experience',)),),
        (registration(required_evidence=('physical',)),), Config(base_relevance=1.))
    assert not g.selected_reasoner_ids


def test_posterior_magnitude_and_highest_identity_do_not_change_routing():
    f = full().multiple_futures
    ids = sorted(b.future_id for b in f.branches)
    values = [1 / sum(range(1, len(ids) + 1)) * i for i in range(1, len(ids) + 1)]
    def belief(vs):
        return BayesianBeliefStateBuilder().initialize(f, priors=dict(zip(ids, vs)), prior_provenance={'source': 'test'})
    a, b = belief(values), belief(list(reversed(values)))
    profiles = (Profile('r', ('belief',), ('belief.posterior.available',)),)
    ga, gb = route(context(belief_state=a), profiles), route(context(belief_state=b), profiles)
    assert ga.relevance_scores == gb.relevance_scores and ga.selected_reasoner_ids == gb.selected_reasoner_ids
    assert ga.context_reference != gb.context_reference
    assert all('posterior_probability' not in d['details'] for d in ga.descriptors)


@pytest.mark.parametrize('status', ['unavailable', 'indeterminate', 'invalid_evidence'])
def test_belief_unknown_categories(status):
    b = BayesianBeliefStateBuilder().initialize(full().multiple_futures)
    b = replace(b, update_status=status, beliefs=tuple(replace(x, update_status=status) for x in b.beliefs))
    ctx = context(belief_state=b)
    assert 'belief.update.' + status in cats(ctx, 'belief')
    assert 'belief.posterior.unavailable' in cats(ctx, 'belief')


def test_experience_enabled_categorical_disabled_no_bonus():
    profiles = (Profile('r', ('experience',), ('experience.positive_context_present',)),)
    enabled = route(experience(), profiles)
    disabled = route(experience(False), profiles)
    assert enabled.relevance_scores['r'] == pytest.approx(.35)
    assert disabled.relevance_scores['r'] == .2


def test_experience_magnitude_not_routing_weight():
    profiles = (Profile('r', ('experience',), ('experience.positive_context_present',)),)
    a, b = route(experience(cap=.001), profiles), route(experience(cap=.05), profiles)
    assert a.relevance_scores == b.relevance_scores and a.selected_reasoner_ids == b.selected_reasoner_ids
    assert all('historical_contribution' not in d['details'] for d in a.descriptors)


def test_profile_blocking_is_explicit_no_unknown_penalty():
    g = route(context(), (Profile('r', blocked_when_flags=('physical.unavailable',)),), config=Config(base_relevance=1.))
    assert g.relevance_scores['r'] == 1.
    assert g.exclusion_reasons['r']['reason'] == 'profile_blocked'


def test_snapshot_audit_edges_and_immutability():
    ctx = full()
    g = route(ctx, (Profile('r', LAYERS),))
    assert json.loads(g.to_json()) == g.to_dict()
    assert g.layer_availability == ctx.layer_availability
    for edge in g.edges:
        assert edge['source_layer'] in LAYERS and edge['routing_rule']
        assert edge['relation'] == 'context_layer_activates_reasoner'
        assert edge['descriptor_refs'] and edge['truth_semantics'] is False
    with pytest.raises(FrozenInstanceError):
        ctx.physical_state = None
    with pytest.raises(TypeError):
        g.relevance_breakdown['r']['layer_bonuses']['belief'] = 1
    with pytest.raises(TypeError):
        g.nodes['r']['enabled'] = False


@pytest.mark.parametrize('change', ['context', 'registry', 'profile', 'config', 'snapshot', 'callback'])
def test_stale_tampered_snapshot_rejected(change):
    ctx, registry, profiles, router = context(), ReasonerRegistry((registration(),)), (Profile('r'),), Router(Config(min_relevance=0.))
    graph = router.route(ctx, registry, profiles)
    if change == 'context':
        ctx = context(item())
    elif change == 'registry':
        registry = ReasonerRegistry((replace(registry.registrations[0], version='new'),))
    elif change == 'profile':
        profiles = (Profile('r', ('evidence',)),)
    elif change == 'config':
        router = Router(Config(min_relevance=.1))
    elif change == 'snapshot':
        graph = replace(graph, relevance_scores={'r': .9})
    else:
        registry = ReasonerRegistry((replace(registry.registrations[0], execute=lambda c: ReasonerOutput()),))
    with pytest.raises(ValueError):
        router.execute(ctx, registry, profiles, graph)


def test_execution_only_orchestrator_selected_and_failed_contained():
    calls = []
    def okay(ctx):
        calls.append('a')
        return ReasonerOutput()
    def failed(ctx):
        calls.append('b')
        raise RuntimeError('contained')
    def never(ctx):
        raise AssertionError('unselected')
    regs = ReasonerRegistry((registration('c', never, enabled=False), registration('b', failed), registration('a', okay)))
    ctx, router = context(), Router(Config(min_relevance=0))
    graph = router.route(ctx, regs)
    assert calls == []
    from fifth_layer.orchestration import ReasonerOrchestrator
    with patch.object(ReasonerOrchestrator, 'plan', autospec=True, wraps=ReasonerOrchestrator.plan) as spy:
        # A wrapping MagicMock cannot substitute plan's tuple; use the actual
        # implementation as side effect while counting eligibility rechecks.
        spy.side_effect = lambda self, c: original_plan(self, c)
        result = router.execute(ctx, regs, (), graph)
    assert spy.call_count >= 2
    assert calls == ['a', 'b']
    assert [r.execution_status for r in result.results] == ['executed', 'failed']
    assert ctx.orchestration_context.common_evidence_state.evidence_items == ()


from fifth_layer.orchestration import ReasonerOrchestrator
original_plan = ReasonerOrchestrator.plan


def test_route_does_not_update_beliefs_or_learning_or_sources():
    ctx = experience()
    before = (ctx.belief_state.to_json(), ctx.experience_view.to_json(), ctx.orchestration_context.common_evidence_state.to_json())
    with (patch('fifth_layer.world_model.bayesian_belief_state.BayesianBeliefStateBuilder.update', side_effect=AssertionError('update')),
          patch('fifth_layer.world_model.experience_learning_v02.experience_learning_v02', side_effect=AssertionError('learning'))):
        g = route(ctx, (Profile('r', ('belief', 'experience')),))
    assert before == (ctx.belief_state.to_json(), ctx.experience_view.to_json(), ctx.orchestration_context.common_evidence_state.to_json())
    assert g.provenance['Bayesian_feedback'] is False and g.provenance['physical_action'] is False


def test_cross_layer_source_mismatch_rejected():
    ctx = full()
    with pytest.raises(ValueError, match='physical source'):
        replace(ctx, physical_state=replace(ctx.physical_state, latent_state_id='wrong'))
    with pytest.raises(ValueError, match='belief/future'):
        replace(ctx, belief_state=replace(ctx.belief_state, source_multiple_futures_id='wrong'), experience_view=None)
    with pytest.raises(ValueError, match='supplied belief'):
        replace(ctx, experience_view=replace(ctx.experience_view, source_belief_state_id='wrong', queries=(), rows=()))


def test_no_clock_required():
    ctx, registry = full(), ReasonerRegistry((registration(),))
    with patch('time.time', side_effect=AssertionError('clock')):
        assert Router().route(ctx, registry).to_json() == Router().route(ctx, registry).to_json()


@pytest.mark.parametrize('name,value', [('scene_id', 'other'), ('session_id', 'other'), ('timestamp', 31.)])
def test_experience_view_current_header_alignment(name, value):
    ctx = experience()
    with pytest.raises(ValueError, match='mismatch'):
        replace(ctx, experience_view=replace(ctx.experience_view, **{name: value}))


def test_history_count_cannot_scale_relevance():
    b = current()
    one = experience_learning_v02(b, trajectory(), enabled=True)
    many = experience_learning_v02(b, trajectory(historical(1), historical(2), historical(3)), enabled=True)
    profiles = (Profile('r', ('experience',), ('experience.positive_context_present',)),)
    a = route(context(belief_state=b, experience_view=one), profiles)
    c = route(context(belief_state=b, experience_view=many), profiles)
    assert a.relevance_scores == c.relevance_scores


def test_experience_evidence_family_ignored_by_routing():
    ctx = context(item('past', 'experience', None))
    g = route(ctx, (Profile('r', ('evidence',), ('evidence.family.experience',)),))
    assert g.relevance_scores['r'] == .2
    assert not g.edges


def test_zero_limit_and_empty_registry():
    assert route(registrations=()).nodes == {}
    graph = route(config=Config(min_relevance=0, max_selected_reasoners=0))
    assert not graph.selected_reasoner_ids and graph.exclusion_reasons['r']['reason'] == 'selection_limit'


def test_physics_uncertainty_retained_not_penalized():
    ctx = full()
    assert ctx.uncertainty['sources']['physics_constraints']['c1'] == ('upstream_uncertainty',)
    assert 'physics_constraints.upstream_uncertainty' in ctx.uncertainty['flags']
    assert route(ctx).relevance_scores['r'] == .2


def test_optional_category_bonus_same_bounded_pool():
    ctx = context(physical_state=moving())
    p = Profile('r', preferred_descriptor_categories=('physical.present',),
                optional_descriptor_categories=('physical.motion_state_present',))
    g = route(ctx, (p,))
    assert g.relevance_breakdown['r']['descriptor_bonus'] == .1
    assert len(g.relevance_breakdown['r']['matched_categories']) == 2


def test_same_registry_metadata_new_callback_invalidates_execution_only():
    ctx, router = context(), Router(Config(min_relevance=0.))
    a = ReasonerRegistry((registration(),))
    b = ReasonerRegistry((registration(),))
    ga, gb = router.route(ctx, a), router.route(ctx, b)
    assert ga.to_json() == gb.to_json()
    with pytest.raises(ValueError, match='binding'):
        router.execute(ctx, b, (), ga)


def test_tracker_identity_not_semantic_relevance():
    p = moving()
    obj = dict(p.objects[0])
    attrs = dict(obj['attributes'])
    attrs['track_id'] = dict(attrs['track_id'], value=999)
    obj['attributes'] = attrs
    changed = replace(p, objects=(obj,))
    profiles = (Profile('r', ('physical',), ('physical.motion_state_present',)),)
    assert route(context(physical_state=p), profiles).relevance_scores == route(context(physical_state=changed), profiles).relevance_scores


@pytest.mark.parametrize('name', ['physical_state', 'physics_constraints', 'cross_modal_consequences',
                                'multiple_futures', 'belief_state', 'experience_view'])
def test_wrong_layer_version_rejected(name):
    ctx = full()
    layer = getattr(ctx, name)
    object.__setattr__(layer, 'schema_version', 'unknown-version')
    with pytest.raises(ValueError):
        replace(ctx)


def test_snapshot_references_change_when_source_content_changes():
    a, b = experience(cap=.001), experience(cap=.05)
    registry, profiles, router = ReasonerRegistry((registration(),)), (Profile('r', ('experience',)),), Router()
    snapshot = router.route(a, registry, profiles)
    with pytest.raises(ValueError, match='stale'):
        router.execute(b, registry, profiles, snapshot)


def test_missing_constraint_and_consequence_links_rejected():
    ctx = full()
    with pytest.raises(ValueError, match='constraint source'):
        replace(ctx, physics_constraints=replace(ctx.physics_constraints, results=()))
    with pytest.raises(ValueError, match='consequence references'):
        replace(ctx, cross_modal_consequences=replace(ctx.cross_modal_consequences, candidates=()))
