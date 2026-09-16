"""Synthetic contract tests; no scientific performance claims from fixtures."""
import ast
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from fifth_layer.confidence_calibration import ConfidenceCalibrationMemory
from fifth_layer.prediction_evaluation import PredictionEvaluationMemory, compare_prediction
from fifth_layer.world_state import WorldState
from fifth_layer.world_model.calibration_assessment import (
    CalibrationContext, FinalizedCalibrationOutcome, adapt_live_prediction,
)
from fifth_layer.world_model.evaluation_metrics import (
    EvaluationUnit, EvaluationMetricsBuilder, EvaluationMetricsReport, MetricEligibility,
    MetricResult, OutcomeDistribution, EvaluationCoverage, adapt_finalized_evaluation,
    represent_unresolved_prediction, probability_metric_eligibility, METRICS, SOURCE_SEMANTICS,
)
from evaluation.tests.test_bayesian_belief_state import initialize, futures
from evaluation.tests.test_spatial_grounding import raw
from evaluation.tests.test_uncertainty_representation import entry
from evaluation.tests.test_reasoning_connectome_v02 import route, experience
from evaluation.tests.test_calibration_assessment import case as calibration_case, B as CALIBRATION
from test_prediction_evaluation import detection

B = EvaluationMetricsBuilder()


def source(status='correct', confidence=.8):
    memory = PredictionEvaluationMemory()
    state = WorldState(0., dict(image_width=600, image_height=800, detections=[detection()]))
    prediction, = memory.record_state(state,
        dict(track_id=1, raw_confidence=confidence, motion_state='stationary',
             trajectory=[dict(center=[100, 100], horizon_seconds=1.)]), 'temporal_live')
    context = CalibrationContext('session', 0., 'scene', 'pixels')
    issuance = adapt_live_prediction(prediction, context=context, source_reference='prediction:' + prediction['prediction_id'])
    if status in ('pending', 'missing_outcome'):
        return memory, prediction, issuance, None
    if status in ('correct', 'partially_correct', 'incorrect', 'image_geometry_changed'):
        x = {'correct': 100, 'partially_correct': 132, 'incorrect': 500, 'image_geometry_changed': 100}[status]
        memory.observe(1., [detection(x)], 601 if status == 'image_geometry_changed' else 600, 800)
    result, = memory.advance(62. if status == 'expired' else 1.25)
    if status == 'image_geometry_changed':
        # Memory prefilters geometry mismatches. The public comparison function
        # reports the explicit reason; this fixture finalizes it at the cutoff.
        result = compare_prediction(prediction, memory.observations[0], 1.25)
    assert result['status'] == ('unevaluable' if status == 'image_geometry_changed' else status)
    return memory, prediction, issuance, result


def unit(status='correct', confidence=.8):
    _, _, issuance, result = source(status, confidence)
    if result is None:
        return represent_unresolved_prediction(issuance, lifecycle=status, source_reference='inventory')
    return adapt_finalized_evaluation(result, issuance=issuance, source_reference='evaluation',
        reference_id='observation:track1:time1' if result['status'] in ('correct', 'partially_correct', 'incorrect') else None,
        finalized=True, provenance={'reference_origin': 'synthetic_observed_track'})


def report(units):
    return B.build('dataset', units, as_of_timestamp=100., session_id='session',
                   assumptions=('supplied_inventory_only',), provenance={'data': 'synthetic'})


def metrics(value):
    return {m.metric_name: m for m in value.metrics}


def test_coverage_distribution_weighting_and_strict_accuracy():
    units = tuple(unit(status) for status in ('correct', 'partially_correct', 'incorrect', 'unevaluable', 'expired', 'pending', 'missing_outcome'))
    r = report(units)
    m = metrics(r)
    assert r.coverage.total_records == 7
    assert r.coverage.evaluable_count == r.coverage.source_evaluable_count == 3
    assert r.coverage.excluded_count == 4
    assert r.coverage.pending_count == r.coverage.missing_outcome_count == 1
    assert r.coverage.unevaluable_count == r.coverage.expired_count == 1
    assert r.outcome_distribution == OutcomeDistribution(1, 1, 1, 3)
    assert m['evaluation_coverage'].value == 3 / 7
    assert m['evaluation_coverage'].denominator == 7
    assert m['weighted_outcome_score'].numerator == 1.5
    assert m['weighted_outcome_score'].denominator == 3
    assert m['weighted_outcome_score'].value == .5
    assert m['strict_accuracy'].value == 1 / 3
    assert m['incorrect_outcome_rate'].value == 1 / 3
    assert m['incorrect_outcome_rate'].value != 1 - m['weighted_outcome_score'].value
    for name in ('correct_outcome_frequency', 'partial_outcome_frequency', 'incorrect_outcome_frequency'):
        assert m[name].value == 1 / 3


@pytest.mark.parametrize('status', ['unevaluable', 'expired', 'pending', 'missing_outcome'])
def test_excluded_records_never_count_as_incorrect(status):
    r = report((unit(status),))
    assert r.coverage.total_records == r.coverage.excluded_count == 1
    assert r.outcome_distribution.incorrect_count == r.outcome_distribution.evaluable_count == 0
    m = metrics(r)
    assert m['evaluation_coverage'].value == 0.  # Valid measured coverage, not a performance placeholder.
    assert m['weighted_outcome_score'].value is None
    assert m['strict_accuracy'].eligibility.status == 'insufficient_data'
    assert r.units[0].exclusion_reasons == (status,)


def test_empty_inventory_has_no_zero_metric_placeholders():
    r = report(())
    assert r.coverage.total_records == 0
    assert all(m.value is None for m in r.metrics)
    assert metrics(r)['evaluation_coverage'].eligibility.status == 'insufficient_data'


def test_partial_only_is_not_strict_success():
    m = metrics(report((unit('partially_correct'),)))
    assert m['weighted_outcome_score'].value == .5
    assert m['strict_accuracy'].value == 0.
    assert m['incorrect_outcome_rate'].value == 0.
    assert m['partial_outcome_frequency'].value == 1.


@pytest.mark.parametrize('metric', ['brier_score', 'nll', 'ece'])
@pytest.mark.parametrize('confidence', [0., .1, .5, .9, 1.])
def test_bounded_confidence_does_not_enable_probability_metrics(metric, confidence):
    r = report((unit(confidence=confidence),))
    result = metrics(r)[metric]
    assert result.value is result.numerator is result.denominator is None
    assert result.sample_count == 0
    assert result.eligibility.status == 'ineligible'
    assert not result.eligibility.eligible
    assert 'contractual_probability_forecast' in result.eligibility.requirements_missing
    assert r.units[0].issuance.calibration['raw_confidence'] == confidence


@pytest.mark.parametrize('factory', [initialize, futures])
@pytest.mark.parametrize('metric', ['brier_score', 'nll', 'ece'])
def test_branch_probability_metric_ineligibility_is_inspectable(factory, metric):
    src = factory()
    before = src.to_json()
    eligibility = probability_metric_eligibility(metric, src)
    assert not eligibility.eligible
    assert eligibility.status == 'ineligible'
    assert 'mutually_exclusive_exhaustive_event_semantics' in eligibility.requirements_missing
    assert eligibility.target_semantics == 'no_resolved_event_target_supplied'
    assert json.loads(eligibility.to_json()) == eligibility.to_dict()
    assert src.to_json() == before


def test_tta_requires_protocol_even_with_timestamps():
    m = metrics(report((unit(),)))['tta']
    assert m.value is None and m.eligibility.status == 'protocol_required'
    assert set(m.eligibility.requirements_missing) == {'event_onset', 'valid_anticipation',
        'anticipation_threshold', 'temporal_alignment_policy', 'false_anticipation_policy', 'censoring_policy'}


@pytest.mark.parametrize('factory', [initialize, futures, raw, entry, route, experience])
def test_unrelated_sources_not_accepted_as_evaluation_units(factory):
    src = factory()
    with pytest.raises(ValueError): report((src,))
    with pytest.raises(ValueError):
        adapt_finalized_evaluation(src, issuance=unit().issuance, source_reference='bad', finalized=True)


@pytest.mark.parametrize('src', [.8, {'posterior_probability': .8}, {'backend_score': .8},
    {'relevance': .8}, {'similarity': .8}, {'uncertainty': .2}, {'probability': .8, 'target': True}])
def test_dicts_and_numbers_cannot_claim_probability_semantics(src):
    with pytest.raises(ValueError): probability_metric_eligibility('brier_score', src)


@pytest.mark.parametrize('name', ['overall_score', 'world_model_score', 'system_quality', 'accuracy', 'p_value'])
def test_no_global_or_unspecified_metric(name):
    with pytest.raises(ValueError): MetricEligibility(name, 1)
    assert name not in report((unit(),)).to_dict()


@pytest.mark.parametrize('kind', SOURCE_SEMANTICS[1:])
def test_branch_labels_cannot_enable_accuracy(kind):
    e = MetricEligibility('strict_accuracy', 10, source_semantics=kind)
    assert not e.eligible and e.status == 'ineligible'


def test_unknown_issuance_is_excluded_without_assuming_chronology():
    u = unit()
    unknown = replace(u, issuance=replace(u.issuance, context=replace(u.issuance.context, timestamp=None)))
    r = report((unknown,))
    assert r.coverage.source_evaluable_count == 1
    assert r.coverage.unqualified_evaluable_count == 1
    assert r.coverage.evaluable_count == 0
    assert unknown.exclusion_reasons == ('issuance_time_unavailable',)
    assert metrics(r)['strict_accuracy'].value is None


def test_missing_reference_and_unknown_time_count_one_excluded_record():
    u = unit()
    u = replace(u, reference_id=None,
        issuance=replace(u.issuance, context=replace(u.issuance.context, timestamp=None)))
    r = report((u,))
    assert len(u.exclusion_reasons) == 2
    assert r.coverage.unqualified_evaluable_count == r.coverage.excluded_count == 1


@pytest.mark.parametrize('time', [1., 1.1, 2.])
def test_future_or_equal_issuance_rejected(time):
    u = unit()
    with pytest.raises(ValueError):
        replace(u, issuance=replace(u.issuance, context=replace(u.issuance.context, timestamp=time)))


def test_evaluation_cannot_precede_target_or_report_cutoff():
    u = unit()
    with pytest.raises(ValueError): replace(u, outcome=dict(u.outcome, evaluated_timestamp=.5))
    with pytest.raises(ValueError): B.build('dataset', (u,), as_of_timestamp=1.)


@pytest.mark.parametrize('time', [0., 2., None])
def test_invalid_reference_chronology_rejected(time):
    u = unit()
    with pytest.raises(ValueError):
        replace(u, outcome=dict(u.outcome, observation=dict(u.outcome['observation'], observed_timestamp=time)))


def test_confidence_cannot_be_retrospectively_rewritten():
    u = unit()
    changed = dict(u.outcome['prediction'], raw_confidence=.1)
    with pytest.raises(ValueError): replace(u, outcome=dict(u.outcome, prediction=changed))


def test_other_prediction_content_cannot_be_rewritten():
    u = unit()
    changed = dict(u.outcome['prediction'], predicted_center=(300, 300))
    with pytest.raises(ValueError): replace(u, outcome=dict(u.outcome, prediction=changed))


def test_geometry_changed_valid_unevaluable_contract_is_accounted():
    u = unit('image_geometry_changed')
    assert u.outcome['evaluation_reason'] == 'image_geometry_changed'
    r = report((u,))
    assert r.coverage.unevaluable_count == 1
    assert r.outcome_distribution.incorrect_count == 0


def test_existing_typed_finalized_outcome_can_be_consumed():
    _, _, issuance, result = source()
    frozen = FinalizedCalibrationOutcome(result)
    before = frozen.to_json()
    u = adapt_finalized_evaluation(frozen, issuance=issuance, source_reference='typed', reference_id='reference', finalized=True)
    assert report((u,)).coverage.evaluable_count == 1
    assert frozen.to_json() == before


def test_public_calibration_assessment_and_uncertainty_sources_unchanged():
    _, _, issuance, stats, outcomes = calibration_case()
    a = CALIBRATION.build(issuance, statistics=stats, outcomes=outcomes)
    e = entry()
    before = a.to_json(), e.to_json()
    report((unit(),))
    assert before == (a.to_json(), e.to_json())
    with pytest.raises(ValueError): report((a,))
    with pytest.raises(ValueError): report((e,))


def test_source_evaluation_and_memory_unchanged():
    memory, p, issuance, outcome = source()
    before = deepcopy(p), deepcopy(outcome), memory.calibration.statistics(), deepcopy(tuple(memory.history))
    with patch.object(ConfidenceCalibrationMemory, 'calibrate', side_effect=AssertionError), \
         patch.object(ConfidenceCalibrationMemory, 'update', side_effect=AssertionError), \
         patch.object(PredictionEvaluationMemory, 'advance', side_effect=AssertionError):
        u = adapt_finalized_evaluation(outcome, issuance=issuance, source_reference='eval', reference_id='ref', finalized=True)
        r = report((u,))
    assert before == (p, outcome, memory.calibration.statistics(), tuple(memory.history))
    assert r.units[0].issuance.calibration == issuance.calibration


def test_deterministic_id_order_and_json_serialization():
    units = (unit(), unit('partially_correct'), unit('expired'))
    a, b = report(units), report(units[::-1])
    assert a.report_id == b.report_id
    assert a.to_json() == b.to_json()
    assert json.loads(a.to_json()) == a.to_dict()
    assert a == replace(a)
    assert a.report_id != replace(a, dataset_id='other').report_id
    assert tuple(m.metric_name for m in a.metrics) == tuple(sorted(METRICS))


def test_nested_immutability_and_detachment():
    _, _, issuance, outcome = source()
    metadata = {'nested': ['original']}
    u = adapt_finalized_evaluation(outcome, issuance=issuance, source_reference='eval', reference_id='ref', finalized=True, provenance=metadata)
    r = report([u])
    metadata['nested'].append('changed')
    outcome['prediction']['predicted_center'].append(42)
    assert u.provenance['caller_provenance']['nested'] == ('original',)
    assert len(u.outcome['prediction']['predicted_center']) == 2
    for obj, name, value in ((r, 'dataset_id', 'x'), (u, 'outcome_eligible', False),
        (r.coverage, 'total_records', 9), (r.outcome_distribution, 'correct_count', 9),
        (r.metrics[0], 'value', .9), (r.metrics[0].eligibility, 'eligible', True)):
        with pytest.raises(FrozenInstanceError): setattr(obj, name, value)
    with pytest.raises(TypeError): u.outcome['status'] = 'incorrect'
    data = r.to_dict()
    data['units'][0]['outcome']['status'] = 'incorrect'
    assert r.units[0].outcome['status'] == 'correct'


def test_denominators_and_contributing_lineage_are_explicit():
    units = (unit(), unit('unevaluable'))
    r = report(units)
    m = metrics(r)
    assert m['evaluation_coverage'].denominator_definition == 'all_supplied_unique_prediction_units'
    assert set(m['evaluation_coverage'].contributing_unit_ids) == {u.unit_id for u in units}
    assert m['weighted_outcome_score'].contributing_unit_ids == (units[0].unit_id,)
    assert units[0].reference_id in m['weighted_outcome_score'].source_references
    assert all(m.eligibility.to_dict() for m in r.metrics)
    assert 'outcome_eligible' in r.to_dict()['units'][0]


@pytest.mark.parametrize('value', [-1, .5, True, float('nan'), float('inf')])
def test_invalid_counts_rejected(value):
    with pytest.raises(ValueError): OutcomeDistribution(value, 0, 0, 0)
    with pytest.raises(ValueError): MetricEligibility('strict_accuracy', value)
    with pytest.raises(ValueError): EvaluationCoverage(value, 0, 0, 0, 0, 0, 0, 0, 0)


def test_inconsistent_counts_and_duplicate_units_rejected():
    with pytest.raises(ValueError): OutcomeDistribution(1, 1, 1, 2)
    with pytest.raises(ValueError): EvaluationCoverage(3, 1, 2, 1, 0, 0, 0, 0, 0)
    u = unit()
    with pytest.raises(ValueError): report((u, u))
    with pytest.raises(ValueError): report((u, replace(u, reference_id='different')))


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_source_metadata_and_metric_numerator_rejected(value):
    u = unit()
    with pytest.raises(ValueError): replace(u, provenance={'nested': value})
    with pytest.raises(ValueError): replace(u, outcome=dict(u.outcome, center_error_pixels=value))
    with pytest.raises(ValueError): replace(metrics(report((u,)))['strict_accuracy'], numerator=value)


def test_ineligible_result_cannot_be_zero_placeholder():
    result = metrics(report((unit(),)))['brier_score']
    with pytest.raises(ValueError): replace(result, numerator=0, denominator=0)
    with pytest.raises(ValueError): replace(result, contributing_unit_ids=('invented',))


@pytest.mark.parametrize('change', [dict(numerator=-1), dict(numerator=2), dict(denominator=0),
    dict(denominator=2), dict(source_references=()), dict(contributing_unit_ids=('x', 'x'))])
def test_metric_result_consistency(change):
    result = metrics(report((unit(),)))['strict_accuracy']
    with pytest.raises(ValueError): replace(result, **change)


@pytest.mark.parametrize('change', [dict(status='pending'), dict(status='missing'), dict(status='unavailable'),
    dict(prediction_id='other'), dict(direction_match=.8), dict(center_error_pixels=None), dict(extra='unsupported')])
def test_invalid_finalized_source_contract(change):
    u = unit()
    with pytest.raises(ValueError): replace(u, outcome=dict(u.outcome, **change))


def test_explicit_finalization_required():
    _, _, issuance, result = source()
    with pytest.raises(ValueError):
        adapt_finalized_evaluation(result, issuance=issuance, source_reference='pending', finalized=False)
    with pytest.raises(ValueError): represent_unresolved_prediction(issuance, lifecycle='correct', source_reference='bad')


def test_session_mismatch_rejected_and_unknown_context_preserved():
    u = unit()
    with pytest.raises(ValueError): B.build('dataset', (u,), session_id='other')
    unknown = replace(u, issuance=replace(u.issuance, context=replace(u.issuance.context, session_id=None, scene_id=None)))
    assert B.build('dataset', (unknown,)).units[0].issuance.context.session_id is None


@pytest.mark.parametrize('value', [b'image', {'tensor': [1]}, 'x' * 16385])
def test_opaque_oversized_metadata_rejected(value):
    with pytest.raises(ValueError): B.build('dataset', (), provenance={'data': value})


def test_no_heavy_imports_and_process_independent_ids():
    code = """
import sys
from fifth_layer.world_model.evaluation_metrics import EvaluationMetricsBuilder
assert not {'numpy', 'torch', 'cv2', 'transformers', 'PIL'} & set(sys.modules)
print(EvaluationMetricsBuilder().build('dataset', ()).to_json())
"""
    outputs = [subprocess.check_output([sys.executable, '-c', code], text=True,
        env=dict(os.environ, PYTHONHASHSEED=seed)) for seed in ('1', '42')]
    assert outputs[0] == outputs[1]


def test_no_inference_fitting_significance_or_io_calls():
    from fifth_layer.world_model import evaluation_metrics as module
    tree = ast.parse(Path(module.__file__).read_bytes())
    calls = {n.func.id if isinstance(n.func, ast.Name) else n.func.attr for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))}
    assert not {'open', 'eval', 'exec', 'calibrate', 'update', 'fit', 'predict', 'random', 'uuid4',
                'bootstrap', 'ttest_ind', 'permutation_test', 'time'} & calls


@pytest.mark.parametrize('change', [dict(observed_center=None), dict(observed_bbox=None),
    dict(observed_center=(1,)), dict(observed_bbox=(3, 3, 1, 1)),
    dict(observation_state='predicted'), dict(is_predicted=True), dict(track_id=99)])
def test_invalid_observation_reference_rejected(change):
    u = unit()
    with pytest.raises(ValueError):
        replace(u, outcome=dict(u.outcome, observation=dict(u.outcome['observation'], **change)))


def test_reference_must_match_target_window():
    u = unit()
    changed = dict(u.outcome, evaluated_timestamp=3.,
        observation=dict(u.outcome['observation'], observed_timestamp=2.))
    with pytest.raises(ValueError, match='target window'): replace(u, outcome=changed)


def test_comparison_before_finalization_window_is_not_finalized_evidence():
    u = unit()
    with pytest.raises(ValueError, match='finalization window'):
        replace(u, outcome=dict(u.outcome, evaluated_timestamp=1.1))


def test_error_measurement_larger_than_one_is_not_probability():
    u = unit('incorrect')
    changed = replace(u, outcome=dict(u.outcome, normalized_center_error=2.5))
    assert report((changed,)).outcome_distribution.incorrect_count == 1
    assert changed.outcome['normalized_center_error'] == 2.5


def test_metadata_depth_cycle_and_node_limits():
    cycle = {}
    cycle['self'] = cycle
    with pytest.raises(ValueError): B.build('dataset', (), provenance=cycle)
    deep = {}
    for _ in range(26): deep = {'child': deep}
    with pytest.raises(ValueError): B.build('dataset', (), provenance=deep)
    with pytest.raises(ValueError): B.build('dataset', (), provenance={'items': [0] * 50001})


def test_no_weighted_averaging_of_confidence_or_reliability():
    a, b = unit('correct', .1), unit('incorrect', .9)
    r = report((a, b))
    assert metrics(r)['weighted_outcome_score'].value == .5
    assert a.issuance.calibration['raw_confidence'] == .1
    assert b.issuance.calibration['raw_confidence'] == .9
    assert all('confidence' not in m.metric_name and 'reliability' not in m.metric_name for m in r.metrics)


def test_conflicting_unresolved_fields_and_untyped_units_rejected():
    u = unit()
    with pytest.raises(ValueError): replace(u, lifecycle='pending')
    with pytest.raises(ValueError): report(({},))
    with pytest.raises(ValueError): report(None)


def test_empty_and_nonempty_denominator_semantics_are_distinct():
    empty = metrics(report(()))['evaluation_coverage']
    nonempty = metrics(report((unit('pending'),)))['evaluation_coverage']
    assert empty.denominator is None and empty.value is None
    assert nonempty.denominator == 1 and nonempty.value == 0.
