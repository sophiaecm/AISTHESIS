"""Synthetic bounded-context contracts; no task-level learning claim."""
from dataclasses import FrozenInstanceError, replace
import inspect
import json
from unittest.mock import patch

import pytest

from fifth_layer.world_model import experience_learning_v02 as learning
from fifth_layer.world_model._structured import freeze
from fifth_layer.world_model.astra_efa_experience import AstraEFAExperienceBuilder as B
from fifth_layer.world_model.bayesian_belief_state import BayesianBeliefStateBuilder
from fifth_layer.world_model.belief_prediction_bridge import issue_prediction, evaluate_issue
from fifth_layer.world_model.experience_memory import ExperienceMemory
from fifth_layer.world_model.multiple_futures import MultipleFuturesBuilder
from fifth_layer.world_model.prediction_observation import experience_episode
from fifth_layer.world_model.prediction_records import PredictionRecord, OutcomeRecord, PredictionEvaluation
from evaluation.tests.test_belief_prediction_bridge import evaluate, sources, select, later
from evaluation.tests.test_multiple_futures import moving

run = learning.experience_learning_v02


def current(timestamp=30., mode='uniform_uninformative'):
    futures = MultipleFuturesBuilder().build(replace(moving(), timestamp=timestamp))
    return BayesianBeliefStateBuilder().initialize(futures, initialization_mode=mode)


def historical(index=1, status='supported', metadata=None, drop=(), outcome_id=None):
    result = evaluate()
    p = result.prediction
    provenance = dict(p.provenance)
    provenance.update(metadata or {})
    for key in drop:
        provenance.pop(key, None)
    p = replace(p, prediction_id=f'past-p{index}', source_timestamp=2. * index,
        target_timestamp=2. * index + 1, provenance=provenance)
    o = replace(result.outcome, prediction_id=p.prediction_id, outcome_id=outcome_id or f'past-o{index}',
        observation_timestamp=p.target_timestamp)
    e = replace(result.evaluation, prediction_id=p.prediction_id, outcome_id=o.outcome_id,
        evaluation_id=f'past-e{index}', status=status, evaluated_timestamp=o.observation_timestamp)
    return experience_episode(p, o, e)


def trajectory(*episodes):
    return B.build(episodes or [historical()], session_id='session', timestamp=0.)


def row(view, state):
    return next(r for r in view.rows if r.hypothesis_id == select(state).hypothesis_id)


def query(state=None):
    state = current() if state is None else state
    return learning.TrajectoryExperienceQuery.from_current(state, select(state).hypothesis_id)


def test_default_disabled_never_reads_history():
    class Unreadable:
        def __getattribute__(self, name):
            raise AssertionError('history read')
    with patch.object(learning.TrajectoryExperienceRetriever, '_retrieve_validated', side_effect=AssertionError('retrieval')):
        view = run(current(), Unreadable())
    assert view.enabled is False and view.source_trajectory_id is None
    assert view.retrieval_audit == () and view.provenance['retrieval_performed'] is False
    assert all(r.historical_contribution == 0 and r.influence_status == 'disabled' for r in view.rows)


@pytest.mark.parametrize('value', [None, {}, (), 'belief'])
def test_belief_type_required(value):
    with pytest.raises(ValueError, match='BayesianBeliefState'):
        run(value)


@pytest.mark.parametrize('value', [None, {}, (), 'trajectory'])
def test_trajectory_required_when_enabled(value):
    with pytest.raises(ValueError, match='AstraExperienceTrajectory'):
        run(current(), value, enabled=True)


@pytest.mark.parametrize('kwargs', [
    {'enabled': 1}, {'top_k': 0}, {'top_k': 4}, {'top_k': True}, {'top_k': 1.5},
    {'min_similarity': -.1}, {'min_similarity': 1.1}, {'min_similarity': True},
    {'min_similarity': float('nan')}, {'max_absolute_contribution': .051},
    {'max_absolute_contribution': -.1}, {'max_absolute_contribution': True},
    {'max_absolute_contribution': float('inf')},
])
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        run(current(), **kwargs)


def test_queries_preserve_structure_exclude_identity_and_probabilities():
    state = current()
    view = run(state)
    assert len(view.queries) == len(state.beliefs)
    for q, b in zip(view.queries, state.beliefs):
        assert q.hypothesis_id == b.hypothesis_id
        assert q.branch_family == b.source_future.branch_family
        assert q.assumption_types == tuple(sorted(a.assumption_type for a in b.source_future.assumptions))
        assert q.provenance['predicted_changes'] == b.source_future.predicted_changes
        assert q.current_structural_features['object_count'] == len(b.source_future.object_ids)
        assert set(q.current_structural_features) == set(learning.FEATURES)
        for forbidden in ('prior_probability', 'posterior_probability', 'future_id', 'scene_id', 'track_id', 'object_id'):
            assert forbidden not in q.current_structural_features


def test_probability_changes_do_not_change_similarity_features():
    a = current()
    futures = MultipleFuturesBuilder().build(replace(moving(), timestamp=30.))
    b = BayesianBeliefStateBuilder().initialize(futures,
        priors=dict(zip(a.source_future_ids, (.1, .9))), prior_provenance={'source': 'synthetic'})
    assert query(a).current_structural_features == query(b).current_structural_features


def test_historical_features_only_stored_structure():
    e = trajectory().experiences[0].encounter
    f = learning.historical_features(e)
    assert f['object_count'] is None
    assert f['branch_family'] == 'motion_continuation'
    assert f['predicted_values'] == (('motion_state', 'moving', 'branch_assumption'),)
    assert f['has_consequence_references'] is False


@pytest.mark.parametrize('timestamp,reason', [(3., 'not_strictly_past'), (2., 'not_strictly_past'), (None, 'unknown_query_timestamp')])
def test_only_strict_past(timestamp, reason):
    state = current(timestamp)
    view = run(state, trajectory(), enabled=True)
    assert all(not r.retrieved_experiences and r.historical_contribution == 0 for r in view.rows)
    assert all(a['checks'][0]['reason'] == reason for a in view.retrieval_audit)


def test_session_mismatch_audited_and_not_retrieved():
    ep = historical()
    # A session-scoped synthetic historical episode with no nested bridge provenance.
    p = dict(ep.prediction_summary, provenance={'session_id': 'other'})
    o = dict(ep.observation_summary, provenance={})
    t = B.build([replace(ep, prediction_summary=p, observation_summary=o)], session_id='other', timestamp=0.)
    view = run(current(), t, enabled=True)
    assert all(a['checks'][0]['reason'] == 'session_mismatch' for a in view.retrieval_audit)
    assert all(r.historical_contribution == 0 for r in view.rows)


@pytest.mark.parametrize('target', ['trajectory_id', 'current_state', 'experiences', 'transformation'])
def test_broken_trajectory_rejected(target):
    t = trajectory()
    if target == 'transformation':
        object.__setattr__(t.experiences[0].transformation, 'transformation_id', 'forged')
    elif target == 'experiences':
        object.__setattr__(t, target, t.experiences * 2)
    else:
        object.__setattr__(t, target, 'forged')
    with pytest.raises(ValueError):
        run(current(), t, enabled=True)


def test_duplicate_outcome_not_independent_evidence():
    t = trajectory(historical(1, outcome_id='same'), historical(2, outcome_id='same'))
    view = run(current(), t, enabled=True)
    r = row(view, current())
    assert len(r.retrieved_experiences) == 1
    assert all(a['checks'][1]['reason'] == 'duplicate_history' for a in view.retrieval_audit)


def test_similarity_exact_and_missing_denominator():
    q = query()
    f = learning.historical_features(trajectory().experiences[0].encounter)
    score, matched, conflicts, unavailable, components, reason = learning.structural_comparison(q.current_structural_features, f)
    assert score == 1 and not conflicts and unavailable == ('object_count',)
    assert len(matched) == 6 and reason == 'structurally_relevant'
    assert components['object_count']['result'] == 'unavailable'


def test_conflicts_lower_similarity_and_threshold_enforced():
    t = trajectory(historical(metadata={'source_horizon_kind': 'short_horizon'}))
    state = current()
    r = row(run(state, t, enabled=True), state)
    assert r.retrieved_experiences[0].similarity_score == pytest.approx(5 / 6)
    assert r.retrieved_experiences[0].conflicting_features == ('horizon_kind',)
    high = row(run(state, t, enabled=True, min_similarity=.9), state)
    assert high.retrieved_experiences == () and high.influence_status == 'no_relevant_history'


@pytest.mark.parametrize('metadata,drop', [
    ({}, ('source_assumptions', 'source_predicted_changes', 'source_horizon_kind', 'source_consequence_ids')),
    ({'source_branch_family': 'contact_resolution'}, ()),
    ({}, ('source_branch_family', 'source_assumptions', 'source_predicted_changes', 'source_horizon_kind', 'source_consequence_ids')),
])
def test_generic_or_missing_features_not_retrieved(metadata, drop):
    state = current()
    r = row(run(state, trajectory(historical(metadata=metadata, drop=drop)), enabled=True, min_similarity=0), state)
    assert not r.retrieved_experiences and r.historical_contribution == 0


def test_no_comparable_fields_zero_similarity():
    empty = dict.fromkeys(learning.FEATURES)
    score, matched, conflicting, unavailable, _, _ = learning.structural_comparison(empty, empty)
    assert score == 0 and matched == conflicting == () and set(unavailable) == set(learning.FEATURES)


@pytest.mark.parametrize('status,direction', list(learning.DIRECTION.items()))
def test_status_direction(status, direction):
    state = current()
    r = row(run(state, trajectory(historical(status=status)), enabled=True), state)
    assert r.historical_contribution == pytest.approx(.05 * r.base_posterior_probability * direction)
    assert r.provenance['directional_terms'] == (direction,)
    assert r.experience_informed_score == r.base_posterior_probability + r.historical_contribution


@pytest.mark.parametrize('cap', [0., .001, .02, .05])
def test_absolute_and_relative_bounds(cap):
    state = current()
    r = row(run(state, trajectory(), enabled=True, max_absolute_contribution=cap), state)
    assert r.historical_contribution == min(cap, .05 * r.base_posterior_probability)


def test_many_history_top_k_deterministic_no_accumulation():
    state = current(100.)
    t = trajectory(*(historical(i) for i in range(1, 9)))
    view = run(state, t, enabled=True)
    r = row(view, state)
    assert len(r.retrieved_experiences) == 3
    assert tuple(x.sequence_index for x in r.retrieved_experiences) == (8, 7, 6)
    assert r.historical_contribution == .05 * r.base_posterior_probability
    checks = next(a['checks'] for a in view.retrieval_audit if a['hypothesis_id'] == r.hypothesis_id)
    assert len(checks) == 8 and sum(a['retrieved'] for a in checks) == 3
    assert run(state, t, enabled=True).to_json() == view.to_json()


def test_directional_mean_not_sum_and_neutral_denominator():
    state = current()
    t = trajectory(historical(1), historical(2, 'contradicted'), historical(3, 'unobservable'))
    r = row(run(state, t, enabled=True), state)
    assert r.historical_contribution == 0
    positive = trajectory(historical(1), historical(2, 'partially_supported'))
    r = row(run(state, positive, enabled=True), state)
    assert r.historical_contribution == pytest.approx(.05 * r.base_posterior_probability / 2)


def test_no_normalization_no_winner():
    state = current()
    view = run(state, trajectory(), enabled=True)
    assert sum(r.base_posterior_probability for r in view.rows) == 1
    assert sum(r.experience_informed_score for r in view.rows) != 1
    assert view.provenance['normalized_distribution'] is False
    assert view.provenance['winner_selected'] is False
    assert tuple(r.hypothesis_id for r in view.rows) == tuple(sorted(r.hypothesis_id for r in view.rows))


@pytest.mark.parametrize('status', ['unavailable', 'indeterminate', 'invalid_evidence'])
def test_failed_belief_never_fabricates_score(status):
    state = current(mode='unavailable')
    state = replace(state, update_status=status, beliefs=tuple(replace(b, update_status=status) for b in state.beliefs))
    view = run(state, trajectory(), enabled=True)
    assert all(r.historical_contribution == 0 and r.experience_informed_score is None for r in view.rows)
    assert all(r.influence_status == 'current_belief_' + status for r in view.rows)


@pytest.mark.parametrize('status', ['unsupported', 'unavailable', 'indeterminate'])
def test_current_future_status_suppresses_history(status):
    state = current()
    state = replace(state, beliefs=tuple(replace(b, source_future=replace(b.source_future, status=status)) for b in state.beliefs))
    view = run(state, trajectory(), enabled=True)
    assert all(r.historical_contribution == 0 and r.influence_status == 'current_future_' + status for r in view.rows)


def test_explicit_source_conflict_suppresses():
    state = current()
    state = replace(state, beliefs=tuple(replace(b, source_future=replace(b.source_future,
        uncertainty=b.source_future.uncertainty + ('source_constraint_conflict_or_ambiguity',))) for b in state.beliefs))
    view = run(state, trajectory(), enabled=True)
    assert all(r.historical_contribution == 0 and r.influence_status == 'current_source_conflict' for r in view.rows)


def test_no_mutation_no_feedback_or_observation_creation():
    state, t = current(), trajectory()
    state_before, trajectory_before = state.to_json(), t.to_json()
    memory = ExperienceMemory(clock=lambda: 0.)
    ep = historical()
    memory.add(ep)
    memory_before = memory.recent()
    with (patch('fifth_layer.world_model.bayesian_belief_state.BayesianBeliefStateBuilder.update', side_effect=AssertionError('update')),
         patch('fifth_layer.world_model.bayesian_belief_state.LikelihoodEvidence.__post_init__', side_effect=AssertionError('likelihood')),
         patch('fifth_layer.world_model.evidence.EvidenceItem.__post_init__', side_effect=AssertionError('evidence')),
         patch('fifth_layer.world_model.experience_learning.experience_context', side_effect=AssertionError('v0.1'))):
        view = run(state, t, enabled=True)
    assert state.to_json() == state_before and t.to_json() == trajectory_before
    assert memory.recent() == memory_before
    assert view.provenance['current_observation'] is False


def test_transformation_and_metric_provenance():
    state, t = current(), trajectory()
    r = row(run(state, t, enabled=True), state).retrieved_experiences[0]
    exp = t.experiences[0]
    assert r.historical_transformation_id == exp.transformation.transformation_id
    assert r.pre_state_id == exp.pre_state.state_id and r.post_state_id == exp.post_state.state_id
    assert r.sequence_index == exp.sequence_index
    assert r.historical_prediction_metrics == exp.encounter.evaluation_summary['metrics']
    assert r.provenance['historical_prediction_provenance'] == exp.encounter.prediction_summary['provenance']
    assert r.provenance['recency_decay'] is False


def test_output_deep_immutability_and_json():
    state = current()
    view = run(state, trajectory(), enabled=True)
    r = row(view, state)
    for obj in (view, view.queries[0], r, r.retrieved_experiences[0]):
        with pytest.raises(FrozenInstanceError):
            obj.schema_version = 'changed'
        with pytest.raises(TypeError):
            obj.provenance['calibrated'] = True
    assert json.loads(view.to_json()) == view.to_dict()
    data = view.to_dict()
    data['provenance']['current_observation'] = True
    assert view.provenance['current_observation'] is False


def test_audit_has_comparison_and_bound_explanation():
    state = current()
    view = run(state, trajectory(historical(1), historical(2)), enabled=True, top_k=1)
    assert all(len(a['checks']) == 2 for a in view.retrieval_audit)
    for audit in view.retrieval_audit:
        for check in audit['checks']:
            assert type(check['eligible']) is bool and check['reason']
            assert set(check['structural_comparison']) == set(learning.FEATURES)
    r = row(view, state)
    assert r.provenance['bound'] == .025 and r.provenance['mean_directional_term'] == 1
    assert r.provenance['formula'] and r.provenance['relative_bound'] == .05


def test_reference_values_never_semantic_similarity():
    t = trajectory(historical(metadata={'source_future_id': 'different', 'source_prior_probability': .01,
        'source_posterior_probability': .99, 'source_constraint_ids': ('different-constraint',)}))
    state = current()
    r = row(run(state, t, enabled=True), state)
    assert r.retrieved_experiences[0].similarity_score == 1.


def test_cross_modal_reference_ids_excluded_from_values():
    a = learning._features('contact_resolution', [{'assumption_type': 'contact_within_horizon'}],
        [{'field': 'contact_status', 'value': 'possible_contact', 'epistemic_status': 'branch_assumption'},
         {'field': 'consequence_reference', 'value': 'one', 'epistemic_status': 'expected'}],
        'next_transition', ['one'])
    b = learning._features('contact_resolution', [{'assumption_type': 'contact_within_horizon'}],
        [{'field': 'contact_status', 'value': 'possible_contact', 'epistemic_status': 'branch_assumption'},
         {'field': 'consequence_reference', 'value': 'two', 'epistemic_status': 'expected'}],
        'next_transition', ['two'])
    assert a == b and a['has_consequence_references'] is True
    assert "'one'" not in str(a) and "'two'" not in str(b)


def test_no_clock_random_or_hidden_inference_dependencies():
    source = inspect.getsource(learning)
    for text in ('import random', 'import time', 'uuid4', 'datetime.now', '.train(', '.fit(', 'EvidenceItem('):
        assert text not in source
    with patch('time.time', side_effect=AssertionError('clock')):
        assert run(current(), trajectory(), enabled=True).to_json() == run(current(), trajectory(), enabled=True).to_json()


@pytest.mark.parametrize('metadata', [
    {'source_assumptions': None, 'source_predicted_changes': None},
    {'source_assumptions': [{'assumption_type': 'hidden_actor'}], 'source_predicted_changes': []},
    {'source_assumptions': [], 'source_predicted_changes': [{'field': 'track_id', 'value': 'moving'}]},
    {'source_assumptions': [], 'source_predicted_changes': [{'field': ['motion_state'], 'value': 'moving'}]},
])
def test_malformed_semantic_features_are_unavailable(metadata):
    state = current()
    r = row(run(state, trajectory(historical(metadata=metadata)), enabled=True), state)
    assert r.retrieved_experiences == () and r.historical_contribution == 0


def test_future_prediction_target_excluded_even_if_evaluated():
    ep = historical(status='insufficient_evidence')
    p = dict(ep.prediction_summary, target_timestamp=40., horizon_seconds=38.)
    t = trajectory(replace(ep, prediction_summary=p))
    view = run(current(), t, enabled=True)
    assert all(a['checks'][0]['reason'] == 'not_strictly_past' for a in view.retrieval_audit)


def test_zero_posterior_no_score_increase():
    futures = MultipleFuturesBuilder().build(replace(moving(), timestamp=30.))
    chosen = select(current()).hypothesis_id
    state = BayesianBeliefStateBuilder().initialize(futures,
        priors={f.future_id: 0. if f.future_id == chosen else 1. for f in futures.branches},
        prior_provenance={'source': 'synthetic'})
    r = row(run(state, trajectory(), enabled=True), state)
    assert r.base_posterior_probability == r.historical_contribution == r.experience_informed_score == 0


def test_empty_history_explicit_no_relevant_history():
    state = current()
    t = B.build([], session_id='session', timestamp=0.)
    r = row(run(state, t, enabled=True), state)
    assert r.influence_status == 'no_relevant_history' and r.historical_contribution == 0


def test_direct_retriever_validates_lineage():
    retrieved, audit = learning.TrajectoryExperienceRetriever().retrieve(query(), trajectory())
    assert len(retrieved) == 1 and audit[0]['reason'] == 'retrieved'
    with pytest.raises(ValueError):
        learning.TrajectoryExperienceRetriever().retrieve(query(), None)


def test_wrong_source_versions_rejected():
    state = current()
    object.__setattr__(state, 'schema_version', 'future-version')
    with pytest.raises(ValueError):
        run(state)
    t = trajectory()
    object.__setattr__(t, 'schema_version', 'future-version')
    with pytest.raises(ValueError):
        run(current(), t, enabled=True)


def test_similarity_precedes_recency_in_top_k():
    state = current()
    t = trajectory(historical(1), historical(2, metadata={'source_horizon_kind': 'short_horizon'}))
    r = row(run(state, t, enabled=True, top_k=1), state)
    assert r.retrieved_experiences[0].sequence_index == 1


@pytest.mark.parametrize('field,value', [('prediction_summary', {}), ('evaluation_status', 'pending')])
def test_incomplete_or_nonterminal_history_rejected(field, value):
    t = trajectory()
    raw = dict(t.experiences[0].encounter.source_episode, **{field: value})
    object.__setattr__(t.experiences[0].encounter, 'source_episode', freeze(raw))
    with pytest.raises(ValueError):
        run(current(), t, enabled=True)


def test_real_track_loss_stays_neutral_history():
    result = evaluate(scene=later(observed_objects=(), motion_evidence=()))
    assert result.status == 'unobservable'
    state = current()
    t = trajectory(result.experience_episode)
    r = row(run(state, t, enabled=True), state)
    assert r.retrieved_experiences[0].historical_status == 'unobservable'
    assert r.historical_contribution == 0
    assert r.retrieved_experiences[0].provenance['current_observation'] is False
    assert t.experiences[0].encounter.observation_summary['observed_state'] == {}


def test_current_future_leakage_rejected():
    state = current()
    b = state.beliefs[0]
    # Frozen APIs allow arbitrary provenance at some nested boundaries; v0.2
    # checks the complete current structure against its declared time.
    object.__setattr__(b, 'provenance', freeze({'source_timestamp': 99.}))
    with pytest.raises(ValueError, match='future'):
        run(state, trajectory(), enabled=True)
