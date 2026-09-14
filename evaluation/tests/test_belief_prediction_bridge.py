"""Offline bridge tests against actual frozen prediction/evaluation functions."""
from dataclasses import FrozenInstanceError, replace
import json
import os
import subprocess
import sys
from unittest.mock import patch

import pytest

from fifth_layer.world_model.belief_prediction_bridge import (
    BeliefPredictionIssue, BeliefPredictionOutcome, issue_prediction, evaluate_issue, VERSION)
from fifth_layer.world_model.bayesian_belief_state import BayesianBeliefStateBuilder
from fifth_layer.world_model.cross_modal_consequences import CrossModalConsequenceBuilder
from fifth_layer.world_model.multiple_futures import MultipleFuturesBuilder
from fifth_layer.world_model.prediction_records import PredictionRecord, OutcomeRecord, PredictionEvaluation
from fifth_layer.world_model.prediction_observation import outcome_from_scene, evaluate_prediction, experience_episode
from fifth_layer.world_model.experience_memory import ExperienceEpisode, ExperienceMemory
from fifth_layer.world_model.scene_state import SceneState
from evaluation.tests.test_multiple_futures import moving, discontinuity
from evaluation.tests.test_cross_modal_consequences import contact, collision
from evaluation.tests.test_bayesian_belief_state import evidence


def sources(physical=None, constraints=None, consequences=None):
    physical = moving() if physical is None else physical
    futures = MultipleFuturesBuilder().build(physical, constraints, consequences)
    belief = BayesianBeliefStateBuilder().initialize(futures, initialization_mode='uniform_uninformative')
    return physical, belief


def select(belief, kind='motion_persists'):
    return next(b for b in belief.beliefs if b.source_future.assumptions[0].assumption_type == kind)


def issued():
    physical, belief = sources()
    return issue_prediction(belief, hypothesis_id=select(belief).hypothesis_id,
        target_timestamp=3., physical_state=physical)


def later(motion='moving_left', **changes):
    values = dict(scene_id='later', timestamp=3., observed_objects=({'track_id': 0, 'class_name': 'glass'},),
        motion_evidence=({'track_id': 0, 'motion_state': motion},),
        provenance={'session_id': 'session', 'coordinate_frame_id': 'pixels'})
    values.update(changes)
    return SceneState(**values)


def evaluate(issue=None, scene=None, **kwargs):
    return evaluate_issue(issued() if issue is None else issue, later() if scene is None else scene,
        **({'session_id': 'session', 'coordinate_frame_id': 'pixels'} | kwargs))


def test_belief_and_explicit_selection_required():
    with pytest.raises(ValueError):
        issue_prediction(None, hypothesis_id='h', target_timestamp=3.)
    _, belief = sources()
    with pytest.raises(TypeError):
        issue_prediction(belief, target_timestamp=3.)
    for name in ('', 'unknown', None):
        with pytest.raises(ValueError):
            issue_prediction(belief, hypothesis_id=name, target_timestamp=3.)


def test_no_automatic_highest_posterior_selection():
    physical, uniform = sources()
    source = MultipleFuturesBuilder().build(physical)
    chosen = select(uniform).hypothesis_id
    belief = BayesianBeliefStateBuilder().initialize(source,
        priors={b.future_id: .01 if b.future_id == chosen else .99 for b in source.branches},
        prior_provenance={'source': 'synthetic'})
    result = issue_prediction(belief, hypothesis_id=chosen, target_timestamp=3., physical_state=physical)
    assert result.hypothesis_id == chosen and result.source_posterior_probability == .01
    assert result.issue_status == 'issued'
    assert result.prediction.confidence is None


@pytest.mark.parametrize('timing', [{'target_timestamp': 1.}, {'target_timestamp': 2.},
    {'target_timestamp': float('inf')}, {'target_timestamp': float('nan')}, {'target_timestamp': True},
    {'horizon_seconds': 0.}, {'horizon_seconds': -1.}, {'horizon_seconds': True},
    {'horizon_seconds': float('inf')}, {'target_timestamp': 3., 'horizon_seconds': 2.}])
def test_invalid_time_rejected(timing):
    _, belief = sources()
    with pytest.raises(ValueError):
        issue_prediction(belief, hypothesis_id=select(belief).hypothesis_id, **timing)


@pytest.mark.parametrize('kind', ['next_transition', 'short_horizon'])
def test_qualitative_horizon_does_not_supply_seconds(kind):
    future = MultipleFuturesBuilder().build(moving(), horizon_kind=kind)
    belief = BayesianBeliefStateBuilder().initialize(future)
    issue = issue_prediction(belief, hypothesis_id=select(belief).hypothesis_id)
    assert issue.issue_status == 'unavailable' and issue.prediction is None
    assert issue.target_timestamp is None
    assert issue.source_belief.source_future.horizon_value is None


@pytest.mark.parametrize('timing', [{'target_timestamp': 3.}, {'horizon_seconds': 1.},
    {'target_timestamp': 3., 'horizon_seconds': 1.}])
def test_explicit_target_or_horizon(timing):
    physical, belief = sources()
    issue = issue_prediction(belief, hypothesis_id=select(belief).hypothesis_id, physical_state=physical, **timing)
    assert issue.issue_status == 'issued'
    assert (issue.prediction.source_timestamp, issue.prediction.target_timestamp, issue.prediction.horizon_seconds) == (2., 3., 1.)


def test_unknown_source_time_nonissued():
    physical, belief = sources(replace(moving(), timestamp=None))
    issue = issue_prediction(belief, hypothesis_id=select(belief).hypothesis_id, target_timestamp=3., physical_state=physical)
    assert issue.issue_status == 'unavailable' and issue.prediction is None


def test_safe_motion_mapping_only():
    issue = issued()
    assert isinstance(issue.prediction, PredictionRecord)
    assert issue.prediction.predicted_state == {'motion_state': 'moving'}
    assert issue.evaluable_fields == ('motion_state',) and issue.unevaluable_fields == ()
    assert issue.prediction.hypothesis_type == 'continued_motion'
    assert issue.prediction.object_id is None and issue.prediction.track_id == 0
    assert issue.prediction.image_size is None and issue.prediction.trajectory_id is None
    assert issue.prediction.confidence is None and issue.prediction.uncertainty is None
    assert issue.prediction.evidence_for == issue.prediction.evidence_against == ()


@pytest.mark.parametrize('kind', ['contact_within_horizon', 'no_contact_within_horizon', 'outcome_unresolved'])
def test_contact_not_mapped_to_event(kind):
    physical, belief = sources(contact())
    issue = issue_prediction(belief, hypothesis_id=select(belief, kind).hypothesis_id,
        target_timestamp=3., physical_state=physical)
    assert issue.issue_status == 'insufficient_prediction_content'
    assert issue.prediction is None and issue.unevaluable_fields == ('contact_status',)


def test_continuity_not_visibility_or_occlusion():
    physical, constraints = discontinuity()
    _, belief = sources(physical, constraints)
    for b in belief.beliefs:
        issue = issue_prediction(belief, hypothesis_id=b.hypothesis_id, target_timestamp=3.)
        assert issue.prediction is None and issue.unevaluable_fields == ('continuity_status',)


def test_consequences_not_sensory_claims_and_refs_preserved():
    physical, constraints = collision()
    consequences = CrossModalConsequenceBuilder().build(physical, constraints)
    _, belief = sources(physical, constraints, consequences)
    chosen = select(belief, 'contact_within_horizon')
    issue = issue_prediction(belief, hypothesis_id=chosen.hypothesis_id, target_timestamp=3.)
    assert issue.prediction is None
    assert issue.unevaluable_fields == ('consequence_reference', 'contact_status')
    assert issue.source_belief.source_future.source_constraint_ids == ('c1',)
    assert set(issue.source_belief.source_future.source_consequence_ids) == {c.consequence_id for c in consequences.candidates}
    assert all(c['epistemic_status'] != 'observed' for c in issue.source_belief.source_future.predicted_changes)


def test_unresolved_motion_is_not_stationary():
    _, belief = sources()
    result = issue_prediction(belief, hypothesis_id=select(belief, 'outcome_unresolved').hypothesis_id, target_timestamp=3.)
    assert result.prediction is None and result.unevaluable_fields == ('motion_state',)


def test_mixed_changes_only_supported_field_issued():
    physical, _ = sources()
    futures = MultipleFuturesBuilder().build(physical)
    chosen = next(b for b in futures.branches if b.assumptions[0].assumption_type == 'motion_persists')
    changed = replace(chosen, predicted_changes=chosen.predicted_changes + (
        {'field': 'continuity_status', 'value': 'unresolved', 'epistemic_status': 'branch_assumption'},))
    futures = replace(futures, branches=tuple(changed if b == chosen else b for b in futures.branches))
    belief = BayesianBeliefStateBuilder().initialize(futures)
    result = issue_prediction(belief, hypothesis_id=chosen.future_id, target_timestamp=3., physical_state=physical)
    assert result.prediction.predicted_state == {'motion_state': 'moving'}
    assert result.unevaluable_fields == ('continuity_status',)


def test_source_lineage_and_context():
    physical, belief = sources()
    belief = BayesianBeliefStateBuilder().update(belief, evidence(belief))
    chosen = select(belief)
    issue = issue_prediction(belief, hypothesis_id=chosen.hypothesis_id, target_timestamp=3., physical_state=physical)
    p = issue.prediction.provenance
    assert issue.belief_state_id == p['belief_state_id'] == belief.belief_state_id
    assert issue.hypothesis_id == p['hypothesis_id'] == chosen.hypothesis_id
    assert issue.source_future_id == p['source_future_id'] == chosen.source_future_id
    assert p['source_branch_family'] == chosen.source_future.branch_family
    assert issue.source_belief.source_future.assumptions == chosen.source_future.assumptions
    assert p['source_physical_state_id'] == physical.latent_state_id
    assert p['source_prior_probability'] == chosen.prior_probability
    assert p['source_posterior_probability'] == chosen.posterior_probability
    assert p['evidence_lineage_ids'] == ('event1',)
    assert p['session_id'] == 'session' and p['coordinate_frame_id'] == 'pixels'
    assert p['source_belief_schema'] == 'bayesian-belief-state-v0.1'
    assert p['source_future_schema'] == 'multiple-futures-0.2'
    assert issue.schema_version == VERSION


def test_latest_belief_cutoff_used_for_issuance():
    physical, belief = sources()
    belief = BayesianBeliefStateBuilder().update(belief, evidence(belief, timestamp=3.), as_of_timestamp=3.)
    issue = issue_prediction(belief, hypothesis_id=select(belief).hypothesis_id, target_timestamp=4., physical_state=physical)
    assert issue.prediction.source_timestamp == 3. and issue.prediction.horizon_seconds == 1.
    assert issue.source_belief.source_future.timestamp == 2.
    with pytest.raises(ValueError):
        issue_prediction(belief, hypothesis_id=select(belief).hypothesis_id, target_timestamp=3.)


def test_no_original_track_lookup_means_unbound_not_guessed():
    _, belief = sources()
    issue = issue_prediction(belief, hypothesis_id=select(belief).hypothesis_id, target_timestamp=3.)
    assert issue.prediction.track_id is None and issue.prediction.object_id is None
    result = evaluate(issue)
    assert result.outcome.association_status == 'unbound'
    assert result.status == 'insufficient_evidence'


@pytest.mark.parametrize('changes', [{'latent_state_id': 'other'}, {'scene_id': 'other'}, {'timestamp': 3.},
    {'session_id': 'other'}, {'coordinate_frame_id': 'other'}, {'provenance': {'source_timestamp': 3.}}])
def test_invalid_or_future_association_source(changes):
    physical, belief = sources()
    with pytest.raises(ValueError):
        issue_prediction(belief, hypothesis_id=select(belief).hypothesis_id, target_timestamp=3.,
            physical_state=replace(physical, **changes))


@pytest.mark.parametrize('changes', [{'value': None}, {'status': 'possible'}, {'derived_from': ()}, {'value': True}])
def test_unknown_or_unsupported_track_is_not_association(changes):
    physical, belief = sources()
    obj = physical.objects[0]
    attr = dict(obj['attributes']['track_id'], **changes)
    physical = replace(physical, objects=(dict(obj, attributes=dict(obj['attributes'], track_id=attr)),))
    issue = issue_prediction(belief, hypothesis_id=select(belief).hypothesis_id, target_timestamp=3., physical_state=physical)
    assert issue.prediction.track_id is None
    assert evaluate(issue).status == 'insufficient_evidence'


def test_duplicate_source_track_not_arbitrarily_selected():
    physical, belief = sources()
    physical = replace(physical, objects=physical.objects + (dict(physical.objects[0], physical_object_id='duplicate'),))
    issue = issue_prediction(belief, hypothesis_id=select(belief).hypothesis_id, target_timestamp=3., physical_state=physical)
    assert issue.prediction.track_id is None
    assert issue.provenance['association']['policy'] == 'unbound_ambiguous_source_track_identity'


@pytest.mark.parametrize('motion,status,match', [('moving', 'supported', True), ('moving_left', 'supported', True),
    ('moving_right', 'supported', True), ('stationary', 'contradicted', False), ('unknown', 'insufficient_evidence', None)])
def test_existing_motion_comparator(motion, status, match):
    issue = issued()
    scene = later(motion)
    result = evaluate(issue, scene)
    assert isinstance(result.outcome, OutcomeRecord) and isinstance(result.evaluation, PredictionEvaluation)
    assert result.status == status
    assert result.evaluation.metrics['motion_state_match'] is match
    expected_outcome = outcome_from_scene(issue.prediction, scene)
    expected_evaluation = evaluate_prediction(issue.prediction, expected_outcome)
    assert result.outcome == expected_outcome and result.evaluation == expected_evaluation
    assert result.experience_episode == experience_episode(issue.prediction, expected_outcome, expected_evaluation)


def test_missing_observation_unobservable_not_occlusion():
    result = evaluate(scene=later(observed_objects=()))
    assert result.status == 'unobservable' and result.outcome.association_status == 'missing'
    assert result.outcome.observed_state == {}
    assert 'track_not_observed; loss_is_not_occlusion' in result.evaluation.reasons


def test_no_later_scene_is_missing_not_contradicted():
    result = evaluate_issue(issued(), None, session_id='session', coordinate_frame_id='pixels')
    assert result.status == 'unobservable'
    assert result.outcome.observation_timestamp is None


def test_missing_motion_insufficient_and_missing_event_not_false():
    result = evaluate(scene=later(motion_evidence=()))
    assert result.status == 'insufficient_evidence'
    assert 'motion_state' not in result.outcome.observed_state
    assert 'events' not in result.outcome.observed_state
    assert result.evaluation.metrics['event_match'] is None


def test_predicted_tracks_are_never_outcome_sources():
    result = evaluate(scene=later(observed_objects=(), motion_evidence=(),
        predicted_tracks=({'track_id': 0, 'motion_state': 'moving', 'timestamp': 100.},)))
    assert result.status == 'unobservable' and result.outcome.observed_state == {}


@pytest.mark.parametrize('flags', [{'observed': False}, {'epistemic_status': 'possible'},
    {'timestamp': 4.}, {'observation_timestamp': 2.}])
def test_nonobserved_or_stale_object_records_not_admitted(flags):
    scene = later(observed_objects=(dict(track_id=0, **flags),))
    assert evaluate(scene=scene).status == 'unobservable'


@pytest.mark.parametrize('flags', [{'is_predicted': True}, {'observed': False},
    {'epistemic_status': 'expected'}, {'timestamp': 4.}, {'observation_state': 'predicted'}])
def test_nonobserved_motion_records_not_admitted(flags):
    result = evaluate(scene=later(motion_evidence=(dict(track_id=0, motion_state='moving', **flags),)))
    assert result.status == 'insufficient_evidence'
    assert 'motion_state' not in result.outcome.observed_state


def test_ambiguous_association_remains_ambiguous():
    result = evaluate(scene=later(observed_objects=({'track_id': 0}, {'track_id': 0})))
    assert result.outcome.association_status == 'ambiguous'
    assert result.status == 'insufficient_evidence'


def test_no_class_nearest_or_untyped_rematching():
    for track in ('0', 1):
        result = evaluate(scene=later(observed_objects=({'track_id': track, 'class_name': 'glass'},)))
        assert result.status == 'unobservable'


def test_conflicting_observed_motion_is_not_contradiction():
    result = evaluate(scene=later(motion_evidence=({'track_id': 0, 'motion_state': 'moving'},
                                                 {'track_id': 0, 'motion_state': 'stationary'})))
    assert result.status == 'insufficient_evidence'


@pytest.mark.parametrize('timestamp', [None, 1., 2.])
def test_later_evidence_strictly_required(timestamp):
    with pytest.raises(ValueError):
        evaluate(scene=later(timestamp=timestamp))


def test_evaluator_tolerance_preserved():
    scene = later(timestamp=3.2)
    result = evaluate(scene=scene)
    assert result.status == 'insufficient_evidence'
    assert 'no_observation_at_target_time' in result.evaluation.reasons
    result = evaluate(scene=scene, time_tolerance_seconds=.2)
    assert result.status == 'supported'
    assert result.evaluation.provenance['time_tolerance_seconds'] == .2
    with pytest.raises(ValueError):
        evaluate(scene=later(timestamp=2.), time_tolerance_seconds=10.)


@pytest.mark.parametrize('kwargs', [{'session_id': 'other'}, {'coordinate_frame_id': 'other'},
    {'time_tolerance_seconds': -1.}, {'position_tolerance_pixels': -1.}])
def test_bad_context_or_policy_rejected(kwargs):
    with pytest.raises(ValueError):
        evaluate(**kwargs)


@pytest.mark.parametrize('metadata', [{'session_id': 'other'}, {'coordinate_frame_id': 'other'}, {'source_timestamp': 4.}])
def test_declared_scene_context_or_future_provenance_rejected(metadata):
    with pytest.raises(ValueError):
        evaluate(scene=later(provenance=metadata))


def test_admitted_observation_future_event_time_rejected():
    with pytest.raises(ValueError):
        evaluate(scene=later(observed_objects=({'track_id': 0, 'event_timestamp': 4.},)))


def test_future_source_provenance_cannot_be_issuance_evidence():
    _, belief = sources()
    selected = select(belief)
    # HypothesisBelief permits free source provenance; the bridge checks the full source boundary.
    selected = replace(selected, provenance={'later_observation_timestamp': 3.})
    belief = replace(belief, beliefs=tuple(selected if b.hypothesis_id == selected.hypothesis_id else b for b in belief.beliefs))
    with pytest.raises(ValueError):
        issue_prediction(belief, hypothesis_id=selected.hypothesis_id, target_timestamp=4.)


def test_legacy_error_is_detached_and_episode_is_existing_contract():
    result = evaluate()
    error = result.prediction_error
    assert error.details['status'] == 'supported'
    error.details['status'] = 'changed externally'
    assert result.prediction_error.details['status'] == 'supported'
    assert result.evaluation.status == 'supported'
    assert isinstance(result.experience_episode, ExperienceEpisode)
    assert result.experience_episode.prediction_id == result.prediction.prediction_id
    assert result.experience_episode.provenance['learning'] is False


@pytest.mark.parametrize('motion', ['moving', 'stationary'])
def test_no_bayesian_or_experience_feedback(motion):
    physical, belief = sources()
    before = belief.to_json()
    issue = issue_prediction(belief, hypothesis_id=select(belief).hypothesis_id, target_timestamp=3., physical_state=physical)
    with (patch.object(BayesianBeliefStateBuilder, 'update', side_effect=AssertionError('Bayesian feedback')),
          patch.object(BayesianBeliefStateBuilder, 'initialize', side_effect=AssertionError('new priors')),
          patch.object(ExperienceMemory, 'add', side_effect=AssertionError('memory write'))):
        result = evaluate(issue, later(motion))
    assert result.status in ('supported', 'contradicted')
    assert belief.to_json() == before


@pytest.mark.parametrize('field,value', [('issue_id', ''), ('belief_state_id', ''), ('source_belief', None),
    ('issue_status', 'true'), ('prediction', None), ('target_timestamp', 2.),
    ('evaluable_fields', ('event',)), ('unevaluable_fields', ('visibility',)), ('provenance', {})])
def test_issue_contract_validation(field, value):
    with pytest.raises(ValueError):
        replace(issued(), **{field: value})


def test_prediction_source_mismatch_and_confidence_rejected():
    issue = issued()
    for changes in ({'confidence': .9}, {'hypothesis_id': 'other'},
                    {'predicted_state': {'motion_state': 'stationary'}}, {'evidence_for': ('assumption',)}):
        with pytest.raises(ValueError):
            replace(issue, prediction=replace(issue.prediction, **changes))


def test_nonissued_prediction_cannot_be_evaluated():
    _, belief = sources()
    issue = issue_prediction(belief, hypothesis_id=select(belief).hypothesis_id)
    with pytest.raises(ValueError):
        evaluate(issue)


def test_outcome_wrapper_rejects_broken_links():
    result = evaluate()
    with pytest.raises(ValueError):
        replace(result, outcome=replace(result.outcome, prediction_id='other'))
    with pytest.raises(ValueError):
        replace(result, evaluation=replace(result.evaluation, outcome_id='other'))
    with pytest.raises(ValueError):
        replace(result, experience_episode=replace(result.experience_episode, prediction_error={}))


def test_immutability_determinism_and_json():
    issue = issued()
    result = evaluate(issue)
    assert issue.to_json() == issued().to_json()
    assert result.to_json() == evaluate(issued()).to_json()
    assert json.loads(issue.to_json()) == issue.to_dict()
    assert json.loads(result.to_json()) == result.to_dict()
    with pytest.raises(FrozenInstanceError):
        issue.issue_status = 'true'
    with pytest.raises(TypeError):
        issue.provenance['truth_decision'] = 'performed'
    with pytest.raises(FrozenInstanceError):
        result.evaluation.status = 'contradicted'


def test_no_semantic_probability_or_truth_shortcuts():
    result = evaluate()
    for key in ('hidden_actor', 'winner', 'truth', 'calibrated_probability'):
        assert key not in result.to_dict() and key not in result.prediction.predicted_state
    assert result.prediction.provenance['truth_decision'] == 'not_performed'
    assert result.evaluation.provenance['interpretation'] == 'target-state compatibility; not probability calibration or causal falsification'


def test_process_determinism_without_models():
    script = '''
import sys
from evaluation.tests.test_belief_prediction_bridge import evaluate
print(evaluate().to_json())
assert not {'torch', 'transformers', 'ultralytics', 'cv2', 'numpy'} & set(sys.modules)
'''
    outputs = [subprocess.run([sys.executable, '-c', script], check=True, text=True, capture_output=True,
        env={**os.environ, 'PYTHONHASHSEED': seed}).stdout for seed in ('1', '42')]
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize('changes', [{'track_id': '0'}, {'track_id': 7}, {'object_id': 'o0'},
                                    {'hypothesis_type': 'hidden_actor'}, {'image_size': (10, 10)}])
def test_issue_does_not_allow_new_binding_or_claims(changes):
    issue = issued()
    with pytest.raises(ValueError):
        replace(issue, prediction=replace(issue.prediction, **changes))


def test_future_prediction_provenance_rejected():
    issue = issued()
    with pytest.raises(ValueError):
        replace(issue, prediction=replace(issue.prediction,
            provenance=dict(issue.prediction.provenance, later_scene_timestamp=3.)))


def test_scene_state_rejects_prediction_in_observed_slot():
    with pytest.raises(ValueError):
        later(observed_objects=({'track_id': 0, 'is_predicted': True},))


def test_predicted_assumptions_are_not_scene_input():
    with pytest.raises(ValueError):
        evaluate(scene=issued().source_belief.source_future)


def test_invalid_association_source_type_rejected():
    _, belief = sources()
    with pytest.raises(ValueError):
        issue_prediction(belief, hypothesis_id=select(belief).hypothesis_id, target_timestamp=3., physical_state=later())


def test_tolerance_does_not_change_original_prediction():
    issue = issued()
    before = issue.to_json()
    evaluate(issue, later(timestamp=3.2), time_tolerance_seconds=.2)
    assert issue.to_json() == before


def test_changed_evaluation_time_rejected():
    result = evaluate()
    changed = replace(result.evaluation, evaluated_timestamp=4.)
    with pytest.raises(ValueError):
        replace(result, evaluation=changed,
            experience_episode=experience_episode(result.prediction, result.outcome, changed))
