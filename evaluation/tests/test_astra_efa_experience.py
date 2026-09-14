"""Offline Step 21 contracts and real Step 20 integration."""
from dataclasses import FrozenInstanceError, fields, replace
import inspect
import json
from unittest.mock import patch

import pytest

from fifth_layer.world_model import astra_efa_experience as astra
from fifth_layer.world_model._structured import freeze
from fifth_layer.world_model.common_evidence_state import bounded_plain
from fifth_layer.world_model.prediction_records import PredictionRecord, OutcomeRecord, PredictionEvaluation
from fifth_layer.world_model.prediction_observation import experience_episode, evaluate_prediction
from fifth_layer.world_model.experience_memory import ExperienceMemory
from evaluation.tests.test_belief_prediction_bridge import evaluate, sources, select, later
from fifth_layer.world_model.belief_prediction_bridge import issue_prediction, evaluate_issue

B = astra.AstraEFAExperienceBuilder


def episode(index=1, status='supported', metrics=None):
    p = PredictionRecord(f'p{index}', f's{index}', f'h{index}', 'continued_motion',
        index * 2., index * 2. + 1, 1., {'motion_state': 'moving'}, track_id=0,
        provenance={'session_id': 'session', 'belief_state_id': 'belief', 'source_future_id': 'future'})
    o = OutcomeRecord(f'o{index}', p.prediction_id, f't{index}', p.target_timestamp,
        'matched', {'motion_state': 'moving'}, track_id=0)
    e = PredictionEvaluation(f'e{index}', p.prediction_id, o.outcome_id, status,
        o.observation_timestamp, {} if metrics is None else metrics)
    return experience_episode(p, o, e)


def initial():
    return B.initialize(session_id='session', timestamp=0.)


def test_initial_deterministic_neutral_and_immutable():
    a = initial()
    assert a == initial() and a.to_json() == initial().to_json()
    assert a.session_id == 'session' and a.sequence_index == 0
    assert a.completed_experience_ids == a.completed_episode_ids == a.completed_prediction_ids == ()
    assert a.last_experience_id is None and a.previous_state_id is None
    assert all(v == 0 for v in a.evaluation_counts.values())
    assert a.cumulative_comparable_predictions == 0 and a.cumulative_error_summary == {}
    with pytest.raises(FrozenInstanceError):
        a.timestamp = 1
    with pytest.raises(TypeError):
        a.evaluation_counts['supported'] = 10


@pytest.mark.parametrize('session,timestamp', [('', 0), (None, 0), ('s', True), ('s', -1), ('s', float('nan'))])
def test_initial_invalid(session, timestamp):
    with pytest.raises(ValueError):
        B.initialize(session_id=session, timestamp=timestamp)


@pytest.mark.parametrize('status', astra.TERMINAL)
def test_terminal_update_semantics(status):
    pre = initial()
    snapshot = pre.to_json()
    encounter = B.encode(episode(status=status))
    post = B.update(pre, encounter)
    assert encounter.evaluation_status == status
    assert post.sequence_index == 1
    assert post.cumulative_comparable_predictions == int(status in astra.COMPARABLE)
    assert post.evaluation_counts == {s: int(s == status) for s in astra.TERMINAL}
    for s in astra.TERMINAL:
        assert getattr(post, 'cumulative_' + s) == int(s == status)
    assert pre.to_json() == snapshot
    assert post.previous_state_id == pre.state_id


@pytest.mark.parametrize('status', ['pending', 'expired', 'unevaluable'])
def test_nonterminal_rejected(status):
    with pytest.raises(ValueError):
        B.encode(episode(status=status))


@pytest.mark.parametrize('name', ['prediction_summary', 'observation_summary', 'prediction_error'])
def test_incomplete_summary_rejected(name):
    ep = episode()
    summary = dict(getattr(ep, name))
    summary.pop('provenance')
    with pytest.raises(ValueError):
        B.encode(replace(ep, **{name: summary}))


@pytest.mark.parametrize('changes', [
    {'prediction_id': 'wrong'}, {'hypothesis_id': 'wrong'}, {'source_scene_id': 'wrong'},
    {'target_scene_id': 'wrong'}, {'trajectory_id': 'wrong'}, {'created_timestamp': 1.},
    {'prediction_timestamp': 1.}, {'observation_timestamp': 4.}, {'evaluated_timestamp': 4.},
    {'evaluation_status': 'contradicted'}, {'evaluated_timestamp': None},
])
def test_inconsistent_episode_rejected(changes):
    with pytest.raises(ValueError):
        B.encode(replace(episode(), **changes))


@pytest.mark.parametrize('summary,key,value', [
    ('observation_summary', 'prediction_id', 'wrong'),
    ('prediction_error', 'outcome_id', 'wrong'),
    ('prediction_error', 'prediction_id', 'wrong'),
    ('observation_summary', 'track_id', 1),
    ('observation_summary', 'track_id', '0'),
])
def test_nested_identity_mismatch(summary, key, value):
    ep = episode()
    with pytest.raises(ValueError):
        B.encode(replace(ep, **{summary: dict(getattr(ep, summary), **{key: value})}))


@pytest.mark.parametrize('observation,evaluation', [(2., 3.), (1., 3.), (4., 3.), (None, None)])
def test_temporal_invalid(observation, evaluation):
    ep = episode()
    with pytest.raises(ValueError):
        B.encode(replace(ep, observation_timestamp=observation, evaluated_timestamp=evaluation,
            observation_summary=dict(ep.observation_summary, observation_timestamp=observation),
            prediction_error=dict(ep.prediction_error, evaluated_timestamp=evaluation)))


@pytest.mark.parametrize('summary,cutoff', [('prediction_summary', 2.), ('observation_summary', 3.), ('prediction_error', 3.)])
def test_nested_future_leakage(summary, cutoff):
    ep = episode()
    data = dict(getattr(ep, summary))
    data['provenance'] = dict(data['provenance'], nested={'source_timestamp': cutoff + 1})
    with pytest.raises(ValueError, match='future'):
        B.encode(replace(ep, **{summary: data}))


@pytest.mark.parametrize('summary', ['prediction_summary', 'observation_summary', 'prediction_error'])
def test_nested_session_mismatch(summary):
    ep = episode()
    data = dict(getattr(ep, summary))
    data['provenance'] = dict(data['provenance'], nested={'session_id': 'other'})
    with pytest.raises(ValueError, match='session'):
        B.encode(replace(ep, **{summary: data}))


def test_missing_session_rejected():
    ep = episode()
    with pytest.raises(ValueError):
        B.encode(replace(ep, prediction_summary=dict(ep.prediction_summary, provenance={})))


@pytest.mark.parametrize('marker', [{'epistemic_status': 'expected'}, {'is_predicted': True}, {'observed': False}])
def test_expected_sources_not_observation(marker):
    ep = episode()
    data = dict(ep.observation_summary, provenance={'sources': [{'record': marker}]})
    with pytest.raises(ValueError, match='observation'):
        B.encode(replace(ep, observation_summary=data))


def test_duplicate_episode_and_prediction():
    ep = episode()
    post = B.update(initial(), B.encode(ep))
    for duplicate in (ep, replace(ep, episode_id='different')):
        with pytest.raises(ValueError, match='duplicate'):
            B.update(post, B.encode(duplicate))


def test_recursive_session_rejected():
    with pytest.raises(ValueError, match='session'):
        B.update(B.initialize(session_id='other', timestamp=0), B.encode(episode()))


@pytest.mark.parametrize('timestamp', [3., 4.])
def test_equal_or_reversed_time_rejected(timestamp):
    with pytest.raises(ValueError, match='strictly'):
        B.update(B.initialize(session_id='session', timestamp=timestamp), B.encode(episode()))


def test_transform_exact_and_deterministic():
    pre, encounter = initial(), B.encode(episode())
    exp = astra.AstraExperienceState(encounter, pre)
    t = exp.transformation
    before, after = pre.to_dict(), exp.post_state.to_dict()
    assert t.changed_fields == tuple(k for k in sorted(before) if before[k] != after[k])
    assert t.unchanged_fields == tuple(k for k in sorted(before) if before[k] == after[k])
    assert t.delta_summary == freeze({k: {'before': before[k], 'after': after[k]} for k in t.changed_fields})
    assert 'session_id' in t.unchanged_fields and 'cumulative_contradicted' in t.unchanged_fields
    assert t.pre_state_id == pre.state_id and t.post_state_id == exp.post_state.state_id
    assert t.encounter_id == encounter.encounter_id
    assert t == astra.TransformationEncoder.encode(pre, encounter, exp.post_state)
    with pytest.raises(ValueError):
        astra.TransformationEncoder.encode(pre, encounter, pre)


def test_metrics_preserved_without_aggregation():
    metrics = {'event_match': False, 'motion_state_match': True, 'position_error_pixels': 4.5,
        'normalized_position_error': None, 'position_match': None, 'visibility_match': None,
        'temporal_error_seconds': -0.5, 'sensory_match': None, 'association_confidence': .8}
    ep = episode(metrics=metrics)
    trajectory = B.build([ep, episode(2, metrics={'position_error_pixels': 0.})], session_id='session', timestamp=0)
    assert trajectory.experiences[0].encounter.evaluation_summary['metrics'] == metrics
    assert trajectory.current_state.cumulative_error_summary == {}
    assert trajectory.experiences[0].encounter.evaluation_summary['metrics']['event_match'] is False
    assert trajectory.experiences[0].encounter.evaluation_summary['metrics']['normalized_position_error'] is None
    metrics['position_error_pixels'] = 100
    assert trajectory.experiences[0].encounter.evaluation_summary['metrics']['position_error_pixels'] == 4.5


def test_replay_prefix_raw_and_experience_histories():
    eps = [episode(1), episode(2, 'contradicted'), episode(3, 'unobservable')]
    first = B.build(eps[:1], session_id='session', timestamp=0)
    snapshot = first.to_json()
    full = B.append(B.append(first, eps[1]), eps[2])
    replay = B.replay(eps, session_id='session', timestamp=0)
    assert full == replay and full.to_json() == replay.to_json()
    assert full.trajectory_id == replay.trajectory_id != first.trajectory_id
    assert full.raw_history == tuple(ep.episode_id for ep in eps)
    assert full.experience_ids == tuple(e.experience_id for e in full.experiences)
    assert full.experience_history == tuple((e.experience_id, e.transformation.transformation_id) for e in full.experiences)
    assert set(full.raw_history).isdisjoint(full.experience_ids)
    assert first.to_json() == snapshot and full.experiences[0] == first.experiences[0]
    assert full.experiences[1].pre_state.state_id == first.current_state.state_id
    with pytest.raises(ValueError):
        B.build(list(reversed(eps)), session_id='session', timestamp=0)


def test_actual_step20_and_provenance_no_mutation():
    physical, belief = sources()
    before = belief.to_json()
    issue = issue_prediction(belief, hypothesis_id=select(belief).hypothesis_id,
        target_timestamp=3., physical_state=physical)
    result = evaluate_issue(issue, later(), session_id='session', coordinate_frame_id='pixels')
    ep = result.experience_episode
    snapshot = bounded_plain(ep)
    memory = ExperienceMemory(clock=lambda: 0)
    memory.add(ep)
    memory_before = memory.recent()
    with patch('fifth_layer.world_model.experience_learning.eligibility', side_effect=AssertionError('learning called')):
        trajectory = B.build([ep], session_id='session', timestamp=0)
    encounter = trajectory.experiences[0].encounter
    assert encounter.source_episode == freeze(ep)
    assert encounter.prediction_id == issue.prediction.prediction_id
    assert encounter.hypothesis_id == issue.hypothesis_id
    assert encounter.source_belief_state_id == belief.belief_state_id
    assert encounter.source_future_id == issue.source_future_id
    assert encounter.source_scene_id == ep.source_scene_id and encounter.target_scene_id == ep.target_scene_id
    assert encounter.prediction_summary['provenance'] == issue.prediction.provenance
    assert belief.to_json() == before and bounded_plain(ep) == snapshot and memory.recent() == memory_before


@pytest.mark.parametrize('scene', [later(observed_objects=(), motion_evidence=()), later(motion_evidence=())])
def test_real_missing_evidence_statuses(scene):
    result = evaluate(scene=scene)
    encounter = B.encode(result.experience_episode)
    assert encounter.evaluation_status in ('unobservable', 'insufficient_evidence')
    post = B.update(initial(), encounter)
    assert post.cumulative_contradicted == post.cumulative_comparable_predictions == 0


def test_empty_trajectory_and_serialization_detached():
    t = B.build([], session_id='session', timestamp=0)
    assert t.current_state == t.initial_state and t.raw_history == t.experience_history == ()
    data = t.to_dict()
    data['initial_state']['provenance']['learning'] = True
    assert t.initial_state.provenance['learning'] is False
    assert json.loads(t.to_json()) == t.to_dict()


def test_no_feedback_or_clock_dependencies():
    source = inspect.getsource(astra)
    for forbidden in ('import time', 'import random', 'uuid4', 'datetime.now',
                      'import experience_learning', 'import bayesian_belief_state',
                      'import multiple_futures', 'LikelihoodEvidence', '.train(', '.fit('):
        assert forbidden not in source
    with patch('time.time', side_effect=AssertionError('wall clock')):
        assert B.build([episode()], session_id='session', timestamp=0).to_json() == B.build(
            [episode()], session_id='session', timestamp=0).to_json()


def test_records_deeply_immutable():
    t = B.build([episode()], session_id='session', timestamp=0)
    exp = t.experiences[0]
    for record in (t, exp, exp.encounter, exp.transformation, exp.post_state):
        with pytest.raises(FrozenInstanceError):
            record.schema_version = 'changed'
        with pytest.raises(TypeError):
            record.provenance['learning'] = True
    with pytest.raises(TypeError):
        exp.encounter.source_episode['prediction_summary']['provenance']['session_id'] = 'other'


def test_broken_trajectory_rejected():
    first = B.build([episode()], session_id='session', timestamp=0)
    second = astra.AstraExperienceState(B.encode(episode(2)), initial())
    with pytest.raises(ValueError, match='lineage'):
        astra.AstraExperienceTrajectory(initial(), (first.experiences[0], second))


def test_epistemic_flags():
    exp = B.build([episode(status='contradicted')], session_id='session', timestamp=0).experiences[0]
    for record in (exp, exp.post_state, exp.transformation, exp.encounter):
        assert record.provenance['ground_truth'] is False
        assert record.provenance['world_transformation'] is False
        assert record.provenance['learning'] is False
        assert record.provenance['inference_feedback'] is False
        assert record.provenance['edis'] == 'deferred'
