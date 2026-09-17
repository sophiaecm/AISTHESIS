"""Synthetic protocol examples only; no Experiment 001 dataset or model outputs."""
from dataclasses import FrozenInstanceError, replace
import json
import os
import subprocess
import sys

import pytest

from evaluation.experiment_001.protocol import (
    AcousticSlot, CONDITIONS, PredictionWindow, ProtocolDefinition, TrialAnnotation,
    assess_trial, load_protocol, match_anticipation, shuffle_assignments,
)


def trial(status='positive', **changes):
    values = dict(sequence_id='synthetic', split='test', media_reference='fixture-media',
        annotation_reference='fixture-annotation', status=status, actor_category='object',
        region_id='left', frame_timestamps=tuple(float(i) for i in range(8)),
        visibility=('occluded',) * 3 + ('visible',) * 5 if status == 'positive' else ('absent',) * 8,
        monitor_start_frame=0, onset_frame=3 if status == 'positive' else None,
        exclusion_reason=None if status in ('positive', 'negative') else status)
    values.update(changes)
    return TrialAnnotation(**values)


def prediction(frame=1, **changes):
    values = dict(prediction_id=f'p{frame}', sequence_id='synthetic', condition='C_FULL',
        cutoff_frame=frame, issued_timestamp=float(frame), evidence_through_timestamp=float(frame),
        horizon_seconds=2., decision='anticipate', actor_category='object', region_id='left',
        source_references=('synthetic-prefix', 'synthetic-output'))
    values.update(changes)
    return PredictionWindow(**values)


def log(*commits, condition='C_FULL'):
    by_frame = {p.cutoff_frame: p for p in commits}
    return tuple(by_frame.get(i, prediction(i, condition=condition, decision='abstain', actor_category=None, region_id=None)) for i in range(6))


def assess(t, predictions):
    return assess_trial(load_protocol(), t, predictions, condition='C_FULL')


def test_protocol_identity_and_serialization_are_content_based():
    a = load_protocol()
    data = json.loads(a.manifest_json)
    b = ProtocolDefinition(json.dumps(dict(reversed(list(data.items()))), indent=4))
    assert a.protocol_id == b.protocol_id and a.to_json() == b.to_json()
    data['production_reference'] = 'different frozen source'
    assert ProtocolDefinition(json.dumps(data)).protocol_id != a.protocol_id
    with pytest.raises(FrozenInstanceError):
        a.manifest_json = '{}'


@pytest.mark.parametrize('field,value', [('horizon_seconds', 3.), ('units', 'frames'),
    ('event_onset_rule', 'model_generated'), ('target_semantics', 'substring'),
    ('probability_metric_eligibility', {'brier_score': 'eligible'}), ('horizon_seconds', float('nan'))])
def test_manifest_cannot_silently_change_implemented_rules(field, value):
    data = json.loads(load_protocol().manifest_json)
    data[field] = value
    with pytest.raises(ValueError): ProtocolDefinition(json.dumps(data))


def test_positive_annotation_with_observed_then_occluded_target():
    t = trial(visibility=('visible', 'occluded', 'partial', 'visible', 'visible', 'visible', 'visible', 'visible'),
        monitor_start_frame=1, observed_before_occlusion_frame=0, occlusion_interval=[1, 2])
    assert t.onset_frame == 3 and t.occlusion_interval == (1, 2)
    with pytest.raises(FrozenInstanceError): t.status = 'negative'
    with pytest.raises(TypeError): t.visibility[0] = 'absent'


@pytest.mark.parametrize('status', ['ambiguous', 'missing_annotation', 'censored'])
def test_unresolved_annotation_never_becomes_negative_or_incorrect(status):
    result = assess(trial(status), log(prediction()))
    assert result.status == 'annotation_unavailable' and result.value is None
    assert all(m.status == 'unavailable' for m in result.matches)


def test_negative_annotation_is_legitimate_false_alarm_evidence():
    result = assess(trial('negative'), log(prediction(1), prediction(2)))
    assert result.status == 'negative_trial' and result.value is None
    assert sum(m.status == 'false_anticipation' for m in result.matches) == 2
    assert result.units == 'seconds'


@pytest.mark.parametrize('changes', [dict(onset_frame=4), dict(onset_frame=7),
    dict(visibility=('occluded',) * 3 + ('visible', 'occluded', 'visible', 'visible', 'visible')),
    dict(visibility=('ambiguous',) + ('occluded',) * 2 + ('visible',) * 5),
    dict(occlusion_interval=(3, 4)), dict(observed_before_occlusion_frame=0)])
def test_objective_onset_and_annotation_lineage_rejected_when_inconsistent(changes):
    with pytest.raises(ValueError): trial(**changes)


def test_missing_annotation_is_not_a_negative_label():
    with pytest.raises(ValueError): trial('negative', visibility=('unannotated',) * 8)
    with pytest.raises(ValueError): trial('negative', visibility=('visible',) * 8)
    with pytest.raises(ValueError): trial('missing_annotation', exclusion_reason=None)


@pytest.mark.parametrize('frame,expected', [(0, 'false_anticipation'), (1, 'valid'),
    (2, 'valid'), (3, 'at_or_after_onset'), (4, 'at_or_after_onset')])
def test_pre_event_and_fixed_horizon_rule(frame, expected):
    m = match_anticipation(load_protocol(), trial(), prediction(frame))
    assert m.status == expected
    assert m.lead_seconds == (3. - frame if expected == 'valid' else None)


def test_earliest_valid_anticipation_not_earliest_vague_or_wrong_prediction():
    result = assess(trial(), log(prediction(0), prediction(1), prediction(2)))
    assert result.value == 2. and result.first_prediction_id == 'p1'
    assert result.matches[0].reason == 'target_outside_issued_horizon'
    assert assess(trial(), tuple(reversed(log(prediction(1), prediction(2))))).to_json() == assess(trial(), log(prediction(1), prediction(2))).to_json()


@pytest.mark.parametrize('change', [dict(actor_category='person'), dict(region_id='right'),
    dict(actor_category='any actor'), dict(region_id='left or right')])
def test_specific_target_matching_no_substrings_or_posthoc_synonyms(change):
    assert match_anticipation(load_protocol(), trial(), prediction(**change)).status == 'false_anticipation'


def test_same_rule_applies_to_every_condition():
    for condition in CONDITIONS:
        assert match_anticipation(load_protocol(), trial(), prediction(condition=condition)).lead_seconds == 2.
        assert match_anticipation(load_protocol(), trial('negative'), prediction(condition=condition)).status == 'false_anticipation'


def test_exact_transport_duplicates_collapse_but_multiple_commitments_rejected():
    predictions = log(prediction())
    assert assess(trial(), predictions + (prediction(),)).to_json() == assess(trial(), predictions).to_json()
    with pytest.raises(ValueError): assess(trial(), predictions + (prediction(prediction_id='other'),))
    with pytest.raises(ValueError): assess(trial(), predictions + (prediction(actor_category='person'),))


def test_missing_log_is_not_an_abstention_or_zero_tta():
    result = assess(trial(), (prediction(),))
    assert result.status == 'incomplete_prediction_log' and result.value is None
    assert result.missing_cutoff_frames == (0, 2, 3, 4, 5)
    complete = assess(trial(), log())
    assert complete.status == 'no_anticipation' and complete.value is None


def test_censored_window_not_false_alarm():
    m = match_anticipation(load_protocol(), trial('negative'), prediction(6))
    assert m.status == 'unavailable' and m.reason == 'right_censored_window' and m.lead_seconds is None


def test_sequence_ending_before_expected_event_is_not_negative_truth():
    assert assess(trial('censored'), log(prediction())).status == 'annotation_unavailable'
    short = trial('negative', frame_timestamps=(0., .5), visibility=('absent', 'absent'))
    result = assess(short, ())
    assert result.status == 'no_eligible_windows' and result.value is None


@pytest.mark.parametrize('change', [dict(issued_timestamp=None), dict(evidence_through_timestamp=None)])
def test_missing_prediction_timestamps_are_unavailable(change):
    assert assess(trial(), log(prediction(**change))).status == 'missing_timestamp'


def test_missing_frame_time_has_no_fabricated_tta():
    t = trial(frame_timestamps=(0., 1., 2., None, 4., 5., 6., 7.))
    result = assess(t, log(prediction()))
    assert result.value is None and result.status == 'missing_timestamp'


@pytest.mark.parametrize('change', [dict(evidence_through_timestamp=2.),
    dict(issued_timestamp=.5), dict(issued_timestamp=True), dict(horizon_seconds=0),
    dict(issued_timestamp=float('nan')), dict(horizon_seconds=float('inf'))])
def test_invalid_forecast_chronology_and_nonfinite_values_rejected(change):
    with pytest.raises(ValueError): match_anticipation(load_protocol(), trial(), prediction(**change))


@pytest.mark.parametrize('change', [dict(horizon_seconds=1.), dict(sequence_id='other'),
    dict(evidence_through_timestamp=.5), dict(cutoff_frame=99)])
def test_fair_prefix_and_horizon_requirements(change):
    with pytest.raises(ValueError): match_anticipation(load_protocol(), trial(), prediction(**change))


@pytest.mark.parametrize('timestamps', [(0., 1., 1., 3., 4., 5., 6., 7.),
    (0., 2., 1., 3., 4., 5., 6., 7.), (0., 1., 2., 3., 4., 5., 6., float('inf'))])
def test_invalid_frame_timeline_rejected(timestamps):
    with pytest.raises(ValueError): trial(frame_timestamps=timestamps)


def test_no_probability_metrics_or_composite_score():
    data = json.loads(load_protocol().manifest_json)
    assert data['probability_metric_eligibility'] == {'brier_score': 'ineligible', 'nll': 'ineligible', 'ece': 'ineligible'}
    assert all('composite' not in x and 'overall' not in x for x in data['metric_identifiers'])
    assert 'secondary_only' in data['weighted_outcome_score']


def test_deterministic_split_and_cutoff_preserving_shuffle():
    slots = tuple(AcousticSlot(seq, split, 1, 1., 'packet-' + seq) for split, seq in
        [('test', 'a'), ('test', 'b'), ('test', 'c'), ('development', 'x'), ('development', 'y')])
    result = shuffle_assignments(slots)
    assert result == shuffle_assignments(slots[::-1])
    for recipient, donor in result:
        assert recipient.sequence_id != donor.sequence_id
        assert (recipient.split, recipient.cutoff_frame, recipient.cutoff_timestamp) == (donor.split, donor.cutoff_frame, donor.cutoff_timestamp)
    assert {a.sequence_id: b.sequence_id for a, b in result} == {'a': 'b', 'b': 'c', 'c': 'a', 'x': 'y', 'y': 'x'}


def test_shuffle_singletons_and_different_cutoffs_cannot_self_assign():
    a = AcousticSlot('a', 'test', 1, 1., 'packet-a')
    b = AcousticSlot('b', 'test', 2, 2., 'packet-b')
    assert shuffle_assignments((a, b)) == ((a, None), (b, None))
    with pytest.raises(ValueError): shuffle_assignments((a, a))


def test_shuffle_rejects_duplicate_timing_and_cross_split_sequence_leakage():
    a = AcousticSlot('a', 'test', 1, 1., 'packet-a')
    with pytest.raises(ValueError): shuffle_assignments((a, replace(a, cutoff_timestamp=2.)))
    with pytest.raises(ValueError): shuffle_assignments((a, replace(a, split='development')))


def test_protocol_pixel_and_confirmation_rules_cannot_be_redefined():
    data = json.loads(load_protocol().manifest_json)
    data['onset_annotation']['consecutive_confirming_frames'] = 1
    with pytest.raises(ValueError): ProtocolDefinition(json.dumps(data))


def test_records_detach_lists_and_unavailable_tta_cannot_be_zero():
    refs = ['prefix', 'output']
    p = prediction(source_references=refs)
    refs.append('future')
    assert p.source_references == ('output', 'prefix')
    result = assess(trial(), log())
    with pytest.raises(ValueError): replace(result, value=0.)
    with pytest.raises(FrozenInstanceError): result.value = 0.
    data = result.to_dict()
    data['matches'].clear()
    assert result.matches


def test_fresh_process_import_boundary_and_protocol_identity():
    code = '''
import sys
from evaluation.experiment_001.protocol import load_protocol
assert not {'PIL', 'numpy', 'torch', 'torchvision', 'cv2', 'transformers'} & set(sys.modules)
assert not any(m.startswith('fifth_layer') for m in sys.modules)
print(load_protocol().to_json())
'''
    outputs = [subprocess.check_output([sys.executable, '-c', code], text=True,
        env=dict(os.environ, PYTHONHASHSEED=seed)) for seed in ('1', '42')]
    assert outputs[0] == outputs[1]
