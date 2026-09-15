"""Synthetic engineering evidence tests; no empirical calibration performance claim."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from fifth_layer.confidence_calibration import ConfidenceCalibrationMemory
from fifth_layer.prediction_evaluation import PredictionEvaluationMemory
from fifth_layer.world_model.calibration_assessment import (
    CalibrationContext, CalibrationIssuance, CalibrationBucketStatistics,
    CalibrationStatisticsSnapshot, CalibrationOutcomeEvidence, FinalizedCalibrationOutcome,
    CalibrationAssessment, CalibrationAssessmentBuilder, adapt_calibration_issuance,
    adapt_live_prediction, adapt_calibration_statistics, adapt_finalized_outcomes,
)
from fifth_layer.world_model.common_evidence_state import bounded_plain
from test_prediction_evaluation import register, detection
from evaluation.tests.test_bayesian_belief_state import initialize
from evaluation.tests.test_spatial_grounding import raw
from evaluation.tests.test_uncertainty_representation import entry


C = CalibrationContext('session', 20., 'scene', 'pixels')
B = CalibrationAssessmentBuilder()


def history(n=5, statuses=None):
    memory = PredictionEvaluationMemory()
    for i in range(n):
        t = float(i * 2)
        register(memory, timestamp=t)
        status = 'correct' if statuses is None else statuses[i]
        if status != 'unevaluable':
            x = {'correct': 100, 'partially_correct': 132, 'incorrect': 500}[status]
            memory.observe(t + 1., [detection(x)], 600, 800)
        result, = memory.advance(t + 1.25)
        assert result['status'] == status
    return memory


def case(n=5, statuses=None):
    memory = history(n, statuses)
    prediction = register(memory, timestamp=20.)
    issuance = adapt_live_prediction(prediction, context=C, source_reference='issued:p', provenance={'test': 'synthetic'})
    stats = adapt_calibration_statistics(memory.calibration.statistics(), context=C, source_reference='statistics:at20')
    outcomes = adapt_finalized_outcomes(tuple(memory.history), context=C, source_reference='finalized:at20', finalized=True)
    return memory, prediction, issuance, stats, outcomes


def claim(calibration=None, **kwargs):
    defaults = dict(context=C, prediction_id='p', prediction_source='temporal_live',
        prediction_type='trajectory_position', class_name='person', source_reference='calibrate:p')
    defaults.update(kwargs)
    return adapt_calibration_issuance(calibration, **defaults)


def test_real_issuance_values_and_matching_empirical_evidence():
    memory, p, issuance, stats, outcomes = case()
    before = deepcopy(p), memory.calibration.statistics(), deepcopy(tuple(memory.history))
    a = B.build(issuance, statistics=stats, outcomes=outcomes, assumptions=('same_session',), provenance={'experiment': 'synthetic'})
    assert a.status == 'assessable'
    assert a.evidence_status == 'matching_bucket_statistics'
    for name in ('raw_confidence', 'calibrated_confidence', 'calibration_reliability', 'calibration_samples'):
        assert a.issuance.calibration[name] == p[name]
    assert a.issuance.calibration['calibration_bucket'] == tuple(p['calibration_bucket'])
    assert a.empirical_statistics['total_evaluable'] == 5
    assert a.outcome_summary['total_evaluable'] == 5
    assert a.minimum_samples == 5
    assert a.assumptions == ('same_session',)
    assert a.provenance['experiment'] == 'synthetic'
    assert before == (p, memory.calibration.statistics(), tuple(memory.history))
    assert 'assessable_not_well_calibrated' in a.reasons


@pytest.mark.parametrize('n,status', [(0, 'insufficient_evidence'), (1, 'insufficient_evidence'),
    (4, 'insufficient_evidence'), (5, 'assessable'), (6, 'assessable')])
def test_existing_engineering_threshold(n, status):
    _, _, issuance, stats, _ = case(n)
    a = B.build(issuance, statistics=stats)
    assert a.status == status
    assert a.issuance.calibration['calibration_samples'] == n
    if n < 5:
        assert a.issuance.calibration['calibration_bucket'] == ('global_prior',)
        assert a.issuance.calibration['calibration_reliability'] == .5
    if n == 0:
        assert a.empirical_statistics is None
        assert a.evidence_status == 'no_empirical_samples'


def test_sample_claim_alone_is_not_empirical_evidence():
    issuance = case()[2]
    a = B.build(issuance)
    assert a.status == 'indeterminate' and a.evidence_status == 'claim_only'
    assert a.empirical_statistics is None and a.outcome_summary is None
    assert B.build(case(0)[2]).status == 'insufficient_evidence'


def test_no_claim_no_fabricated_confidence_or_samples():
    a = B.build(claim())
    assert a.status == 'unavailable'
    assert a.issuance.calibration is None
    assert a.empirical_statistics is None
    assert a.outcome_summary is None


def test_partial_outcomes_preserve_half_weight():
    _, _, issuance, stats, outcomes = case(5, ['partially_correct'] * 5)
    a = B.build(issuance, statistics=stats, outcomes=outcomes)
    assert a.status == 'assessable'
    for counts in (a.empirical_statistics, a.outcome_summary):
        assert counts['correct_count'] == 0
        assert counts['partial_count'] == 5
        assert counts['incorrect_count'] == 0
        assert counts['weighted_success'] == 2.5


def test_unevaluable_is_not_incorrect_or_a_sample():
    _, _, issuance, stats, outcomes = case(5, ['unevaluable'] * 5)
    a = B.build(issuance, statistics=stats, outcomes=outcomes)
    assert a.status == 'insufficient_evidence'
    assert a.outcome_summary['unevaluable_count'] == 5
    assert a.outcome_summary['incorrect_count'] == a.outcome_summary['total_evaluable'] == 0


def test_expired_is_not_incorrect():
    memory = PredictionEvaluationMemory()
    register(memory)
    expired, = memory.advance(62.)
    assert expired['status'] == 'expired'
    context = replace(C, timestamp=70.)
    output = memory.calibration.calibrate(.8, 'temporal_live', 'trajectory_position', 'person', 70.)
    issuance = claim(output, context=context)
    evidence = adapt_finalized_outcomes([expired], context=context, source_reference='expired', finalized=True)
    a = B.build(issuance, outcomes=evidence)
    assert a.outcome_summary['expired_count'] == 1
    assert a.outcome_summary['incorrect_count'] == a.outcome_summary['total_evaluable'] == 0


def test_missing_outcomes_differ_from_explicit_empty_subset():
    _, _, issuance, stats, outcomes = case()
    absent = B.build(issuance, statistics=stats)
    empty = B.build(issuance, statistics=stats, outcomes=replace(outcomes, outcomes=()))
    assert absent.outcome_summary is None
    assert empty.outcome_summary['total_evaluable'] == 0
    assert empty.status == 'assessable'  # Supplied statistics remain separate evidence.
    assert empty.outcome_summary['coverage'] == 'supplied_subset_not_memory_membership'


def test_outcomes_not_added_to_statistics_or_used_as_membership_proof():
    _, _, issuance, stats, outcomes = case()
    with_both = B.build(issuance, statistics=stats, outcomes=outcomes)
    assert with_both.empirical_statistics['total_evaluable'] == 5
    assert with_both.issuance.calibration['calibration_samples'] == 5
    with_outcomes = B.build(issuance, outcomes=outcomes)
    assert with_outcomes.status == 'indeterminate'
    assert with_outcomes.evidence_status == 'outcome_subset_only'


@pytest.mark.parametrize('specificity', [1, 2, 3])
def test_actual_fallback_bucket_preserved(specificity):
    memory = ConfidenceCalibrationMemory()
    source_results = list(history().history)
    for i, r in enumerate(source_results):
        r = deepcopy(r)
        if specificity < 3: r['prediction']['class_name'] = str(i)
        if specificity < 2: r['prediction']['prediction_type'] = str(i)
        memory.update(r)
    output = memory.calibrate(.8, 'temporal_live', 'trajectory_position', 'person', 20.)
    assert len(output['calibration_bucket']) == specificity
    stats = adapt_calibration_statistics(memory.statistics(), context=C, source_reference='stats')
    a = B.build(claim(output), statistics=stats)
    assert a.status == 'assessable'
    assert a.empirical_statistics['bucket'] == tuple(output['calibration_bucket'])
    assert a.empirical_statistics['reliability'] == output['calibration_reliability']
    assert 'bucket_specificity_' + {1: 'source', 2: 'prediction_type', 3: 'class'}[specificity] in a.reasons


def test_unrelated_buckets_not_averaged_or_substituted():
    _, _, issuance, stats, outcomes = case()
    unrelated = replace(stats.rows[0], bucket=('other_source',), reliability=.01)
    expanded = replace(stats, rows=(*stats.rows, unrelated))
    a = B.build(issuance, statistics=expanded, outcomes=outcomes)
    assert a.status == 'assessable'
    assert a.empirical_statistics['reliability'] == issuance.calibration['calibration_reliability']
    absent = B.build(issuance, statistics=replace(stats, rows=(unrelated,)))
    assert absent.status == 'indeterminate' and absent.evidence_status == 'bucket_unavailable'


@pytest.mark.parametrize('field', ['calibration_reliability', 'calibration_samples'])
def test_claim_statistics_disagreement_preserved_as_indeterminate(field):
    _, _, issuance, stats, _ = case()
    changed = dict(issuance.calibration, **{field: .1 if field == 'calibration_reliability' else 6})
    a = B.build(replace(issuance, calibration=changed), statistics=stats)
    assert a.status == 'indeterminate' and a.evidence_status == 'conflicting_sources'
    assert a.issuance.calibration[field] == changed[field]


def test_no_inversion_recalibration_or_metric_fields():
    _, p, issuance, stats, _ = case()
    a = B.build(issuance, statistics=stats)
    assert a.issuance.calibration['raw_confidence'] == p['raw_confidence']
    assert a.issuance.calibration['calibration_reliability'] == p['calibration_reliability']
    assert not {'probability', 'uncertainty', 'overall_score', 'ece', 'brier_score', 'nll', 'accuracy'} & a.to_dict().keys()
    assert a.temporal_mode == 'issuance'


@pytest.mark.parametrize('factory', [initialize, raw, entry])
def test_bayesian_grounding_and_uncertainty_are_not_calibration_inputs(factory):
    source = factory()
    with pytest.raises(ValueError): claim(source)
    with pytest.raises(ValueError): B.build(source)
    with pytest.raises(ValueError): adapt_live_prediction(source, context=C, source_reference='bad')
    with pytest.raises(ValueError): adapt_calibration_statistics(source, context=C, source_reference='bad')


@pytest.mark.parametrize('field', ['raw_confidence', 'calibrated_confidence', 'calibration_reliability'])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'), -.1, 1.1, True])
def test_invalid_confidence_and_reliability(field, value):
    output = dict(case(0)[2].calibration)
    output[field] = value
    with pytest.raises(ValueError): claim(output)


@pytest.mark.parametrize('value', [-1, .5, True, float('nan'), float('inf')])
def test_invalid_sample_counts(value):
    output = dict(case(0)[2].calibration, calibration_samples=value)
    with pytest.raises(ValueError): claim(output)


@pytest.mark.parametrize('field,value', [('correct_count', -1), ('partial_count', True),
    ('incorrect_count', .5), ('total_evaluable', 6), ('weighted_success', 2.5),
    ('reliability', 1.1), ('last_updated_timestamp', float('nan')), ('bucket', ('global_prior',))])
def test_statistics_reject_inconsistent_or_invalid_fields(field, value):
    row = case()[3].rows[0]
    with pytest.raises(ValueError): replace(row, **{field: value})


def test_weighted_success_not_bounded_to_one():
    assert case()[3].rows[0].weighted_success == 5.


@pytest.mark.parametrize('value', [None, {}, {'posterior_probability': .8}, {'backend_score': .8}])
def test_wrong_live_prediction_contract(value):
    with pytest.raises(ValueError): adapt_live_prediction(value, context=C, source_reference='bad')


@pytest.mark.parametrize('change', [dict(calibration_bucket=['other']), dict(calibration_bucket=[]),
    dict(calibration_bucket=['temporal_live', 'other_type']), dict(calibration_bucket=['temporal_live', 'trajectory_position', 'other_class']),
    dict(calibration_bucket=['global_prior']), dict(calibration_samples=4)])
def test_inconsistent_issuance_bucket_rejected(change):
    output = dict(case()[2].calibration, **change)
    with pytest.raises(ValueError): claim(output)


def test_future_snapshot_rejected_without_rewriting_issuance():
    _, p, issuance, stats, _ = case()
    before = deepcopy(p)
    with pytest.raises(ValueError, match='future'):
        B.build(issuance, statistics=replace(stats, context=replace(C, timestamp=21.)))
    assert p == before


def test_future_outcome_cannot_support_issuance_even_with_unknown_capture():
    _, _, issuance, _, outcomes = case()
    o = outcomes.outcomes[0]
    future = replace(o, result=dict(o.result, evaluated_timestamp=21.))
    unknown_capture = replace(outcomes, context=replace(C, timestamp=None), outcomes=(future,))
    with pytest.raises(ValueError, match='future'): B.build(issuance, outcomes=unknown_capture)


def test_future_bucket_update_rejected_even_with_unknown_capture():
    _, _, issuance, stats, _ = case()
    stats = replace(stats, context=replace(C, timestamp=None), rows=(replace(stats.rows[0], last_updated_timestamp=21.),))
    with pytest.raises(ValueError, match='future'): B.build(issuance, statistics=stats)


@pytest.mark.parametrize('kind', ['statistics', 'outcomes'])
def test_cross_session_evidence_rejected(kind):
    _, _, issuance, stats, outcomes = case()
    source = stats if kind == 'statistics' else outcomes
    with pytest.raises(ValueError, match='cross-session'):
        B.build(issuance, **{kind: replace(source, context=replace(C, session_id='other'))})


@pytest.mark.parametrize('field', ['timestamp', 'session_id'])
def test_unknown_evidence_context_is_indeterminate(field):
    _, _, issuance, stats, _ = case()
    a = B.build(issuance, statistics=replace(stats, context=replace(C, **{field: None})))
    assert a.status == 'indeterminate' and a.evidence_status == 'context_unverified'


def test_earlier_statistics_not_claimed_as_issuance_snapshot():
    _, _, issuance, stats, _ = case()
    a = B.build(issuance, statistics=replace(stats, context=replace(C, timestamp=19.)))
    assert a.status == 'indeterminate' and a.evidence_status == 'earlier_snapshot'


def test_issuance_time_is_distinct_from_older_prediction_creation_time():
    _, p, _, stats, _ = case()
    p = dict(p, prediction_created_timestamp=0., prediction_horizon_seconds=21.)
    issuance = adapt_live_prediction(p, context=C, source_reference='delayed_issuance')
    a = B.build(issuance, statistics=stats)
    assert a.status == 'assessable'
    assert a.issuance.context.timestamp == 20.
    assert a.issuance.prediction_created_timestamp == 0.


def test_later_memory_updates_do_not_change_issued_values():
    memory, p, issuance, stats, _ = case()
    original = B.build(issuance, statistics=stats).to_json()
    for r in history(5, ['incorrect'] * 5).history:
        r = deepcopy(r)
        r['evaluated_timestamp'] += 30.
        memory.calibration.update(r)
    assert B.build(issuance, statistics=stats).to_json() == original
    assert issuance.calibration['calibrated_confidence'] == p['calibrated_confidence']


def test_current_prediction_cannot_be_its_own_historical_evidence():
    _, _, issuance, _, outcomes = case()
    r = deepcopy(bounded_plain(outcomes.outcomes[0].result))
    r['prediction_id'] = r['prediction']['prediction_id'] = issuance.prediction_id
    evidence = replace(outcomes, outcomes=(FinalizedCalibrationOutcome(r),))
    with pytest.raises(ValueError, match='own historical'): B.build(issuance, outcomes=evidence)


@pytest.mark.parametrize('status', ['pending', 'unavailable', 'correctness_probability'])
def test_nonexistent_finalized_statuses_rejected(status):
    r = dict(case()[4].outcomes[0].result, status=status)
    with pytest.raises(ValueError): FinalizedCalibrationOutcome(r)


def test_finalization_declaration_required():
    with pytest.raises(ValueError):
        adapt_finalized_outcomes([], context=C, source_reference='pending', finalized=False)


def test_duplicate_outcomes_and_bucket_rows_rejected():
    _, _, _, stats, outcomes = case()
    with pytest.raises(ValueError): replace(stats, rows=(stats.rows[0], stats.rows[0]))
    with pytest.raises(ValueError): replace(outcomes, outcomes=(outcomes.outcomes[0], outcomes.outcomes[0]))


def test_evidence_cannot_precede_its_sources():
    _, _, _, stats, outcomes = case()
    with pytest.raises(ValueError): replace(stats, context=replace(C, timestamp=0.))
    with pytest.raises(ValueError): replace(outcomes, context=replace(C, timestamp=0.))


def test_determinism_permutation_and_serialization():
    _, _, issuance, stats, outcomes = case()
    a = B.build(issuance, statistics=stats, outcomes=outcomes, assumptions=('b', 'a'))
    b = B.build(issuance, statistics=replace(stats, rows=stats.rows[::-1]),
        outcomes=replace(outcomes, outcomes=outcomes.outcomes[::-1]), assumptions=('a', 'b'))
    assert a.assessment_id == b.assessment_id
    assert a.to_json() == b.to_json()
    assert json.loads(a.to_json()) == a.to_dict()
    assert a == replace(a)
    assert B.build(issuance, statistics=stats, assumptions=('other',)).assessment_id != a.assessment_id


def test_frozen_and_detached_records():
    memory, p, issuance, stats, outcomes = case()
    source_stats = memory.calibration.statistics()
    copied_stats = adapt_calibration_statistics(source_stats, context=C, source_reference='copied')
    metadata = {'nested': ['original']}
    a = B.build(issuance, statistics=copied_stats, provenance=metadata)
    metadata['nested'].append('changed')
    p['raw_confidence'] = 0.
    source_stats[0]['correct_count'] = 99
    assert a.provenance['nested'] == ('original',)
    assert a.issuance.calibration['raw_confidence'] != 0.
    assert a.statistics.rows[0].correct_count == 5
    for record, field, value in ((a, 'status', 'well_calibrated'), (issuance, 'prediction_id', 'x'),
        (stats, 'rows', ()), (outcomes, 'outcomes', ()), (C, 'timestamp', 0.)):
        with pytest.raises(FrozenInstanceError): setattr(record, field, value)
    with pytest.raises(TypeError): a.empirical_statistics['total_evaluable'] = 99
    serialized = a.to_dict()
    serialized['issuance']['calibration']['raw_confidence'] = 0.
    assert a.issuance.calibration['raw_confidence'] != 0.


def test_does_not_call_mutable_calibration_memory():
    memory, _, issuance, stats, outcomes = case()
    before = deepcopy(memory.calibration.buckets), deepcopy(memory.calibration.processed), memory.calibration.now
    with patch.object(ConfidenceCalibrationMemory, 'calibrate', side_effect=AssertionError), \
         patch.object(ConfidenceCalibrationMemory, 'update', side_effect=AssertionError), \
         patch.object(ConfidenceCalibrationMemory, 'advance', side_effect=AssertionError), \
         patch.object(ConfidenceCalibrationMemory, 'statistics', side_effect=AssertionError):
        B.build(issuance, statistics=stats, outcomes=outcomes)
    assert before == (memory.calibration.buckets, memory.calibration.processed, memory.calibration.now)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), b'raw', {'tensor': [1]}, 'x' * 16385])
def test_invalid_nested_metadata(value):
    with pytest.raises(ValueError): claim(provenance={'nested': value})


def test_bounded_metadata_depth_and_cycles():
    cyclic = {}
    cyclic['cycle'] = cyclic
    with pytest.raises(ValueError): claim(provenance=cyclic)
    deep = {}
    for _ in range(26): deep = {'child': deep}
    with pytest.raises(ValueError): claim(provenance=deep)


def test_no_heavy_imports_or_nondeterministic_identity():
    code = """
import sys
from fifth_layer.world_model.calibration_assessment import *
assert not {'numpy', 'torch', 'cv2', 'transformers', 'PIL'} & set(sys.modules)
c = adapt_calibration_issuance(None, context=CalibrationContext(), prediction_id='p', prediction_source='s', prediction_type='t', source_reference='r')
print(CalibrationAssessmentBuilder().build(c).to_json())
"""
    outputs = [subprocess.check_output([sys.executable, '-c', code], text=True,
        env=dict(os.environ, PYTHONHASHSEED=seed)) for seed in ('1', '42')]
    assert outputs[0] == outputs[1]


def test_no_io_fitting_or_producer_calls_in_production():
    from fifth_layer.world_model import calibration_assessment as module
    tree = ast.parse(Path(module.__file__).read_bytes())
    forbidden = {'open', 'exec', 'eval', 'uuid4', 'random', 'time', 'mean', 'fit', 'calibrate', 'update', 'download'}
    calls = {n.func.id if isinstance(n.func, ast.Name) else n.func.attr for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))}
    assert not forbidden & calls


@pytest.mark.parametrize('changes', [dict(observation=None), dict(center_error_pixels=None),
    dict(evaluation_reason='no_suitable_observation'), dict(direction_match=.5),
    dict(bbox_iou=2.), dict(normalized_center_error=-1.)])
def test_evaluable_outcome_requires_consistent_source_evidence(changes):
    result = case()[4].outcomes[0].result
    with pytest.raises(ValueError): FinalizedCalibrationOutcome(dict(result, **changes))


@pytest.mark.parametrize('change', [dict(observation_state='predicted'), dict(is_predicted=True),
    dict(track_id='other'), dict(observed_timestamp=100.), dict(observed_timestamp=None)])
def test_invalid_observation_cannot_be_counted(change):
    result = case()[4].outcomes[0].result
    with pytest.raises(ValueError):
        FinalizedCalibrationOutcome(dict(result, observation=dict(result['observation'], **change)))


def test_normalized_error_not_assumed_probability_range():
    result = case()[4].outcomes[0].result
    copied = FinalizedCalibrationOutcome(dict(result, normalized_center_error=2.))
    assert copied.result['normalized_center_error'] == 2.


def test_outside_bucket_outcomes_do_not_enter_counts():
    _, _, issuance, stats, outcomes = case()
    result = bounded_plain(outcomes.outcomes[0].result)
    result['prediction']['class_name'] = 'different_class'
    other = FinalizedCalibrationOutcome(result)
    a = B.build(issuance, statistics=stats, outcomes=replace(outcomes, outcomes=(other,)))
    assert a.outcome_summary['outside_bucket_count'] == 1
    assert a.outcome_summary['total_evaluable'] == 0
    assert a.empirical_statistics['total_evaluable'] == 5


def test_prior_and_empirical_bucket_reliability_are_distinct():
    _, _, issuance, stats, _ = case(1)
    a = B.build(issuance, statistics=stats)
    assert a.issuance.calibration['calibration_reliability'] == .5
    assert a.empirical_statistics['reliability'] == .6
    assert a.empirical_statistics['bucket'] == ('temporal_live',)
    assert a.status == 'insufficient_evidence'


def test_unknown_issuance_time_remains_explicit():
    _, _, issuance, stats, _ = case()
    a = B.build(replace(issuance, context=replace(C, timestamp=None)), statistics=stats)
    assert a.status == 'indeterminate'
    assert a.issuance.context.timestamp is None
    assert a.issuance.prediction_created_timestamp == 20.


def test_new_producer_fields_fail_explicitly():
    source = dict(case()[1], posterior_probability=.8)
    with pytest.raises(ValueError): adapt_live_prediction(source, context=C, source_reference='extra')
    rows = [dict(case()[3].rows[0].to_dict(), extra=1)]
    with pytest.raises(ValueError): adapt_calibration_statistics(rows, context=C, source_reference='extra')


def test_unversioned_source_attribution_is_distinct_from_new_schema():
    _, _, issuance, stats, outcomes = case()
    a = B.build(issuance, statistics=stats, outcomes=outcomes)
    assert stats.source_contract == 'ConfidenceCalibrationMemory.statistics'
    assert outcomes.finalization == 'caller_attested_finalized_output'
    assert issuance.context.attribution == 'caller_supplied'
    assert {'issued:p', 'statistics:at20', 'finalized:at20'} <= set(a.source_references)


@pytest.mark.parametrize('outcome_status', ['correct', 'incorrect'])
@pytest.mark.parametrize('confidence', [0., .1, .9, 1.])
def test_assessability_is_not_accuracy_or_a_confidence_threshold(outcome_status, confidence):
    memory = history(5, [outcome_status] * 5).calibration
    output = memory.calibrate(confidence, 'temporal_live', 'trajectory_position', 'person', 20.)
    stats = adapt_calibration_statistics(memory.statistics(), context=C, source_reference='stats')
    a = B.build(claim(output), statistics=stats)
    assert a.status == 'assessable'
    assert a.issuance.calibration['raw_confidence'] == confidence
    assert a.empirical_statistics['correct_count'] == (5 if outcome_status == 'correct' else 0)
