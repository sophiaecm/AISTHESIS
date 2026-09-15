"""Issuance-oriented calibration evidence assessment; no fitting or feedback."""
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
import json
from math import isclose

from ._structured import freeze, identifier, number
from .common_evidence_state import bounded_plain
from .evidence import stable_id
from ..confidence_calibration import MIN_CALIBRATION_SAMPLES, OUTCOME_WEIGHTS

VERSION = 'calibration-assessment-v0.1'
POLICY = 'issuance_bucket_evidence_minimum_v0.1'
STATUSES = ('assessable', 'insufficient_evidence', 'unavailable', 'indeterminate')
FINAL_STATUSES = ('correct', 'partially_correct', 'incorrect', 'unevaluable', 'expired')
_CALIBRATION_FIELDS = frozenset(('raw_confidence', 'calibrated_confidence',
    'calibration_reliability', 'calibration_samples', 'calibration_bucket'))
_LIVE_FIELDS = frozenset(('prediction_id', 'track_id', 'class_name', 'prediction_created_timestamp',
    'prediction_horizon_seconds', 'target_timestamp', 'position_uncertainty_at_prediction',
    'source', 'origin_center', 'image_size', 'predicted_center', 'predicted_bbox',
    'predicted_motion_direction', 'prediction_type')) | _CALIBRATION_FIELDS
_RESULT_FIELDS = frozenset(('prediction_id', 'track_id', 'prediction_created_timestamp',
    'prediction_horizon_seconds', 'target_timestamp', 'source', 'status', 'center_error_pixels',
    'normalized_center_error', 'bbox_iou', 'direction_match', 'timing_error_seconds',
    'evaluated_timestamp', 'evaluation_reason', 'prediction', 'observation'))
_WEIGHTS = freeze(dict(OUTCOME_WEIGHTS))


def _mapping(value):
    if not isinstance(value, Mapping):
        raise ValueError('explicit structured mapping required')
    return freeze(bounded_plain(value))


def _exact(value, expected):
    result = _mapping(value)
    if set(result) != set(expected):
        raise ValueError('unexpected or missing source contract fields')
    return result


def _strings(value):
    if not isinstance(value, (tuple, list)):
        raise ValueError('ordered reference sequence required')
    for item in value:
        identifier(item, 'reference')
    return tuple(sorted(set(value)))


def _count(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(name + ' must be a nonnegative integer')
    number(value, name)


def _bucket(value, *, allow_prior=False):
    if not isinstance(value, (tuple, list)) or not 1 <= len(value) <= 3:
        raise ValueError('source/type/class bucket required')
    for i, part in enumerate(value):
        identifier(part, 'bucket', optional=i == 2)
    result = tuple(value)
    if result == ('global_prior',) and not allow_prior:
        raise ValueError('global prior is not an empirical bucket')
    return result


def _specificity(bucket):
    return 'global_prior' if bucket == ('global_prior',) else {1: 'source', 2: 'prediction_type', 3: 'class'}[len(bucket)]


class _Record:
    def to_dict(self):
        return bounded_plain(self)

    def to_json(self):
        return json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'), allow_nan=False)


def _identity(record, name):
    data = record.to_dict()
    data.pop(name)
    object.__setattr__(record, name, stable_id(name, VERSION, data))


@dataclass(frozen=True)
class CalibrationContext(_Record):
    """Caller attribution: timestamp is capture/issuance time, not forecast origin."""
    session_id: str | None = None
    timestamp: float | None = None
    scene_id: str | None = None
    coordinate_frame_id: str | None = None
    attribution: str = field(default='caller_supplied', init=False)

    def __post_init__(self):
        for name in ('session_id', 'scene_id', 'coordinate_frame_id'):
            identifier(getattr(self, name), name, optional=True)
        if self.timestamp is not None:
            number(self.timestamp, 'timestamp', nonnegative=True)
        bounded_plain(self)


@dataclass(frozen=True)
class CalibrationIssuance(_Record):
    context: CalibrationContext
    prediction_id: str
    prediction_source: str
    prediction_type: str
    class_name: str | None
    calibration: Mapping | None
    source_reference: str
    provenance: Mapping = field(default_factory=dict)
    prediction_created_timestamp: float | None = None
    issuance_id: str = field(default='', init=False)
    schema_version: str = field(default=VERSION, init=False)
    source_contract: str = field(default='ConfidenceCalibrationMemory.calibrate', init=False)

    def __post_init__(self):
        if type(self.context) is not CalibrationContext:
            raise ValueError('CalibrationContext required')
        for name in ('prediction_id', 'prediction_source', 'prediction_type', 'source_reference'):
            identifier(getattr(self, name), name)
        identifier(self.class_name, 'class_name', optional=True)
        if self.prediction_created_timestamp is not None:
            number(self.prediction_created_timestamp, 'prediction_created_timestamp', nonnegative=True)
            if self.context.timestamp is not None and self.prediction_created_timestamp > self.context.timestamp:
                raise ValueError('forecast origin cannot follow issuance')
        if self.calibration is not None:
            data = _exact(self.calibration, _CALIBRATION_FIELDS)
            for name in ('raw_confidence', 'calibrated_confidence', 'calibration_reliability'):
                number(data[name], name, unit=True)
            _count(data['calibration_samples'], 'calibration_samples')
            bucket = _bucket(data['calibration_bucket'], allow_prior=True)
            if bucket != ('global_prior',):
                population = (self.prediction_source, self.prediction_type, self.class_name)
                if bucket != population[:len(bucket)]:
                    raise ValueError('calibration bucket disagrees with prediction population')
                if data['calibration_samples'] < MIN_CALIBRATION_SAMPLES:
                    raise ValueError('empirical issuance bucket below producer activation threshold')
            elif data['calibration_samples'] >= MIN_CALIBRATION_SAMPLES:
                raise ValueError('global prior with sufficient source samples contradicts producer contract')
            object.__setattr__(self, 'calibration', data)
        object.__setattr__(self, 'provenance', _mapping(self.provenance))
        _identity(self, 'issuance_id')


def adapt_calibration_issuance(calibration, *, context, prediction_id, prediction_source,
                               prediction_type, source_reference, class_name=None, provenance=None):
    """Exact calibrate() output (or explicitly absent output) plus caller identity."""
    return CalibrationIssuance(context, prediction_id, prediction_source, prediction_type,
        class_name, calibration, source_reference, {} if provenance is None else provenance)


def _live_prediction(source):
    data = _exact(source, _LIVE_FIELDS)
    for name in ('prediction_id', 'source', 'prediction_type'):
        identifier(data[name], name)
    identifier(data['class_name'], 'class_name', optional=True)
    for name in ('prediction_created_timestamp', 'prediction_horizon_seconds', 'target_timestamp'):
        number(data[name], name, nonnegative=True)
    if not isclose(data['prediction_created_timestamp'] + data['prediction_horizon_seconds'],
                   data['target_timestamp'], abs_tol=1e-9, rel_tol=0.):
        raise ValueError('inconsistent forecast timing')
    return data


def adapt_live_prediction(source, *, context, source_reference, provenance=None):
    """PredictionEvaluationMemory.record_state issuance; context is caller supplied."""
    data = _live_prediction(source)
    return CalibrationIssuance(context, data['prediction_id'], data['source'], data['prediction_type'],
        data['class_name'], {k: data[k] for k in _CALIBRATION_FIELDS}, source_reference,
        {'source_contract': 'PredictionEvaluationMemory.record_state',
         'source_prediction': data, 'caller_provenance': {} if provenance is None else provenance},
        data['prediction_created_timestamp'])


@dataclass(frozen=True)
class CalibrationBucketStatistics(_Record):
    bucket: tuple
    total_evaluable: int
    correct_count: int
    partial_count: int
    incorrect_count: int
    weighted_success: float
    reliability: float
    last_updated_timestamp: float

    def __post_init__(self):
        object.__setattr__(self, 'bucket', _bucket(self.bucket))
        for name in ('total_evaluable', 'correct_count', 'partial_count', 'incorrect_count'):
            _count(getattr(self, name), name)
        if self.total_evaluable != self.correct_count + self.partial_count + self.incorrect_count:
            raise ValueError('inconsistent outcome counts')
        number(self.weighted_success, 'weighted_success', nonnegative=True)
        expected = (self.correct_count * _WEIGHTS['correct'] +
                    self.partial_count * _WEIGHTS['partially_correct'] +
                    self.incorrect_count * _WEIGHTS['incorrect'])
        if self.weighted_success != expected:
            raise ValueError('weighted_success disagrees with existing producer outcome weights')
        number(self.reliability, 'reliability', unit=True)
        number(self.last_updated_timestamp, 'last_updated_timestamp', nonnegative=True)
        bounded_plain(self)


@dataclass(frozen=True)
class CalibrationStatisticsSnapshot(_Record):
    context: CalibrationContext
    rows: tuple[CalibrationBucketStatistics, ...]
    source_reference: str
    provenance: Mapping = field(default_factory=dict)
    snapshot_id: str = field(default='', init=False)
    schema_version: str = field(default=VERSION, init=False)
    source_contract: str = field(default='ConfidenceCalibrationMemory.statistics', init=False)

    def __post_init__(self):
        if type(self.context) is not CalibrationContext:
            raise ValueError('CalibrationContext required')
        identifier(self.source_reference, 'source_reference')
        if not isinstance(self.rows, (tuple, list)) or any(type(r) is not CalibrationBucketStatistics for r in self.rows):
            raise ValueError('typed statistics rows required')
        if len({r.bucket for r in self.rows}) != len(self.rows):
            raise ValueError('duplicate statistics bucket')
        if self.context.timestamp is not None and any(r.last_updated_timestamp > self.context.timestamp for r in self.rows):
            raise ValueError('statistics updated after snapshot capture')
        object.__setattr__(self, 'rows', tuple(sorted(self.rows, key=lambda r: json.dumps(r.bucket))))
        object.__setattr__(self, 'provenance', _mapping(self.provenance))
        _identity(self, 'snapshot_id')


def adapt_calibration_statistics(source, *, context, source_reference, provenance=None):
    """Detached public statistics() result; no memory reads or cleanup calls."""
    if not isinstance(source, (tuple, list)):
        raise ValueError('explicit statistics sequence required')
    bounded_plain(source)
    keys = {f.name for f in fields(CalibrationBucketStatistics)}
    rows = tuple(CalibrationBucketStatistics(**_exact(row, keys)) for row in source)
    return CalibrationStatisticsSnapshot(context, rows, source_reference, {} if provenance is None else provenance)


@dataclass(frozen=True)
class FinalizedCalibrationOutcome(_Record):
    """A supplied finalized evaluation result, not a newly evaluated prediction."""
    result: Mapping
    outcome_id: str = field(default='', init=False)
    schema_version: str = field(default=VERSION, init=False)

    def __post_init__(self):
        result = _exact(self.result, _RESULT_FIELDS)
        prediction = _live_prediction(result['prediction'])
        if result['status'] not in FINAL_STATUSES:
            raise ValueError('only existing finalized evaluation statuses supported')
        for name in ('prediction_id', 'track_id', 'prediction_created_timestamp',
                     'prediction_horizon_seconds', 'target_timestamp', 'source'):
            if result[name] != prediction[name]:
                raise ValueError('outcome/prediction lineage mismatch: ' + name)
        number(result['evaluated_timestamp'], 'evaluated_timestamp', nonnegative=True)
        if result['evaluated_timestamp'] < prediction['target_timestamp']:
            raise ValueError('outcome cannot precede forecast target')
        for name in ('center_error_pixels', 'normalized_center_error', 'bbox_iou', 'timing_error_seconds'):
            value = result[name]
            if value is not None:
                number(value, name, unit=name == 'bbox_iou', nonnegative=name != 'timing_error_seconds')
        if result['direction_match'] is not None and type(result['direction_match']) is not bool:
            raise ValueError('direction_match must be boolean or unavailable')
        if result['status'] in _WEIGHTS:
            if result['evaluation_reason'] != 'same_track_observed_geometry' or any(
                    result[name] is None for name in ('center_error_pixels', 'normalized_center_error',
                                                     'bbox_iou', 'timing_error_seconds', 'direction_match')):
                raise ValueError('evaluable outcome requires supplied evaluation measurements')
            observation = result['observation']
            if not isinstance(observation, Mapping) or observation.get('observation_state') != 'observed' or observation.get('is_predicted'):
                raise ValueError('evaluable outcome requires observed source evidence')
            if observation.get('track_id') != prediction['track_id']:
                raise ValueError('evaluable outcome requires matching observed track')
            observed_time = observation.get('observed_timestamp')
            number(observed_time, 'observed_timestamp', nonnegative=True)
            if not prediction['prediction_created_timestamp'] < observed_time <= result['evaluated_timestamp']:
                raise ValueError('invalid observed/evaluated temporal lineage')
        elif result['status'] == 'unevaluable':
            if result['evaluation_reason'] != 'no_suitable_observation' or any(
                    result[name] is not None for name in ('center_error_pixels', 'normalized_center_error',
                                                        'bbox_iou', 'timing_error_seconds', 'direction_match')):
                raise ValueError('unevaluable source cannot contain evaluated geometry results')
        elif result['evaluation_reason'] != 'memory_limit_or_ttl':
            raise ValueError('expired outcome requires original expiration reason')
        # Validate issued calibration metadata but never recalibrate that prediction.
        CalibrationIssuance(CalibrationContext(), prediction['prediction_id'], prediction['source'],
            prediction['prediction_type'], prediction['class_name'],
            {k: prediction[k] for k in _CALIBRATION_FIELDS}, 'embedded_prediction')
        object.__setattr__(self, 'result', result)
        _identity(self, 'outcome_id')


@dataclass(frozen=True)
class CalibrationOutcomeEvidence(_Record):
    context: CalibrationContext
    outcomes: tuple[FinalizedCalibrationOutcome, ...]
    source_reference: str
    provenance: Mapping = field(default_factory=dict)
    coverage: str = field(default='supplied_subset_not_memory_membership', init=False)
    finalization: str = field(default='caller_attested_finalized_output', init=False)
    source_contract: str = field(default='PredictionEvaluationMemory.finalized_results', init=False)
    evidence_id: str = field(default='', init=False)
    schema_version: str = field(default=VERSION, init=False)

    def __post_init__(self):
        if type(self.context) is not CalibrationContext:
            raise ValueError('CalibrationContext required')
        identifier(self.source_reference, 'source_reference')
        if not isinstance(self.outcomes, (tuple, list)) or any(type(o) is not FinalizedCalibrationOutcome for o in self.outcomes):
            raise ValueError('typed finalized outcomes required')
        ids = [o.result['prediction_id'] for o in self.outcomes]
        if len(ids) != len(set(ids)):
            raise ValueError('duplicate finalized prediction outcome')
        if self.context.timestamp is not None and any(o.result['evaluated_timestamp'] > self.context.timestamp for o in self.outcomes):
            raise ValueError('outcome finalized after evidence capture')
        object.__setattr__(self, 'outcomes', tuple(sorted(self.outcomes, key=lambda o: o.outcome_id)))
        object.__setattr__(self, 'provenance', _mapping(self.provenance))
        _identity(self, 'evidence_id')


def adapt_finalized_outcomes(source, *, context, source_reference, finalized, provenance=None):
    """Caller explicitly attests history/results_for_source outputs are finalized."""
    if finalized is not True or not isinstance(source, (tuple, list)):
        raise ValueError('explicit finalized outcome sequence required')
    bounded_plain(source)
    return CalibrationOutcomeEvidence(context, tuple(FinalizedCalibrationOutcome(r) for r in source),
        source_reference, {} if provenance is None else provenance)


def _evidence_alignment(evidence, issuance):
    a, b = evidence.context, issuance.context
    if a.session_id is not None and b.session_id is not None and a.session_id != b.session_id:
        raise ValueError('cross-session calibration evidence forbidden')
    times = ([r.last_updated_timestamp for r in evidence.rows] if type(evidence) is CalibrationStatisticsSnapshot
             else [o.result['evaluated_timestamp'] for o in evidence.outcomes])
    if a.timestamp is not None:
        times.append(a.timestamp)
    if b.timestamp is not None and any(t > b.timestamp for t in times):
        raise ValueError('future calibration evidence cannot support issuance')
    return a.session_id is not None and b.session_id is not None and a.timestamp is not None and b.timestamp is not None


def _outcome_summary(evidence, bucket):
    counts = dict(total_evaluable=0, correct_count=0, partial_count=0, incorrect_count=0,
                  weighted_success=0., unevaluable_count=0, expired_count=0, outside_bucket_count=0)
    included, excluded = [], []
    for o in evidence.outcomes:
        p, status = o.result['prediction'], o.result['status']
        population = (p['source'], p['prediction_type'], p['class_name'])
        if bucket is None or population[:len(bucket)] != bucket:
            counts['outside_bucket_count'] += 1
            excluded.append(o.outcome_id)
            continue
        if status in _WEIGHTS:
            counts['total_evaluable'] += 1
            counts[{'correct': 'correct_count', 'partially_correct': 'partial_count', 'incorrect': 'incorrect_count'}[status]] += 1
            counts['weighted_success'] += _WEIGHTS[status]
            included.append(o.outcome_id)
        else:
            counts[status + '_count'] += 1
            excluded.append(o.outcome_id)
    return dict(**counts, bucket=bucket, included_outcome_ids=sorted(included), excluded_outcome_ids=sorted(excluded),
        coverage=evidence.coverage, derivation='count_supplied_finalized_results_in_named_bucket',
        outcome_weights=_WEIGHTS, memory_membership_verified=False)


@dataclass(frozen=True)
class CalibrationAssessment(_Record):
    issuance: CalibrationIssuance
    statistics: CalibrationStatisticsSnapshot | None = None
    outcomes: CalibrationOutcomeEvidence | None = None
    assumptions: tuple = ()
    provenance: Mapping = field(default_factory=dict)
    status: str = field(default='', init=False)
    evidence_status: str = field(default='', init=False)
    reasons: tuple = field(default=(), init=False)
    empirical_statistics: Mapping | None = field(default=None, init=False)
    outcome_summary: Mapping | None = field(default=None, init=False)
    source_references: tuple = field(default=(), init=False)
    assessment_id: str = field(default='', init=False)
    assessment_policy: str = field(default=POLICY, init=False)
    minimum_samples: int = field(default=MIN_CALIBRATION_SAMPLES, init=False)
    temporal_mode: str = field(default='issuance', init=False)
    schema_version: str = field(default=VERSION, init=False)
    rule_version: str = field(default=VERSION, init=False)

    def __post_init__(self):
        if type(self.issuance) is not CalibrationIssuance:
            raise ValueError('explicit CalibrationIssuance required')
        for value, cls in ((self.statistics, CalibrationStatisticsSnapshot), (self.outcomes, CalibrationOutcomeEvidence)):
            if value is not None and type(value) is not cls:
                raise ValueError('typed calibration evidence required')
        object.__setattr__(self, 'assumptions', _strings(self.assumptions))
        object.__setattr__(self, 'provenance', _mapping(self.provenance))
        issuance, cal = self.issuance, self.issuance.calibration
        reasons = {'assessable_not_well_calibrated', 'reliability_not_truth_probability',
                   'caller_attributed_context', 'no_recalibration_or_metrics'}
        aligned = True
        for evidence in (self.statistics, self.outcomes):
            if evidence is not None:
                aligned = _evidence_alignment(evidence, issuance) and aligned
        if self.outcomes is not None and any(o.result['prediction_id'] == issuance.prediction_id for o in self.outcomes.outcomes):
            raise ValueError('issued prediction cannot be its own historical calibration evidence')
        bucket = None if cal is None else tuple(cal['calibration_bucket'])
        prior = bucket == ('global_prior',)
        # Below threshold, samples refer to the source bucket, not the fixed prior.
        empirical_bucket = (issuance.prediction_source,) if prior else bucket
        row = None if self.statistics is None else next((r for r in self.statistics.rows if r.bucket == empirical_bucket), None)
        if row is not None:
            object.__setattr__(self, 'empirical_statistics', _mapping(row.to_dict()))
        if self.outcomes is not None:
            object.__setattr__(self, 'outcome_summary', _mapping(_outcome_summary(self.outcomes, empirical_bucket)))
            reasons.add('outcome_subset_not_proof_of_memory_membership')
        if cal is None:
            status, evidence_status = 'unavailable', 'issuance_metadata_unavailable'
            reasons.add('no_calibration_claim_supplied')
        elif self.statistics is None:
            status = 'insufficient_evidence' if cal['calibration_samples'] < self.minimum_samples else 'indeterminate'
            evidence_status = 'claim_only' if self.outcomes is None else 'outcome_subset_only'
            reasons.add('source_statistics_not_supplied')
        elif not aligned:
            status, evidence_status = 'indeterminate', 'context_unverified'
            reasons.add('session_or_capture_time_unknown')
        elif self.statistics.context.timestamp != issuance.context.timestamp:
            status, evidence_status = 'indeterminate', 'earlier_snapshot'
            reasons.add('earlier_snapshot_does_not_establish_issuance_state')
        elif row is None:
            if prior and cal['calibration_samples'] == 0:
                status, evidence_status = 'insufficient_evidence', 'no_empirical_samples'
                reasons.add('fixed_prior_not_empirical_reliability')
            else:
                status, evidence_status = 'indeterminate', 'bucket_unavailable'
                reasons.add('claimed_bucket_or_sample_evidence_missing')
        elif row.total_evaluable != cal['calibration_samples'] or (
                not prior and row.reliability != cal['calibration_reliability']):
            status, evidence_status = 'indeterminate', 'conflicting_sources'
            reasons.add('issuance_and_statistics_disagree')
        elif prior:
            status, evidence_status = 'insufficient_evidence', 'below_activation_threshold'
            reasons.add('fixed_prior_not_empirical_reliability')
        else:
            status, evidence_status = 'assessable', 'matching_bucket_statistics'
            reasons.add('existing_engineering_minimum_met')
        if bucket is not None:
            reasons.add('bucket_specificity_' + _specificity(bucket))
        refs = [issuance.source_reference, issuance.issuance_id]
        for evidence in (self.statistics, self.outcomes):
            if evidence is not None:
                refs.append(evidence.source_reference)
                refs.append(evidence.snapshot_id if type(evidence) is CalibrationStatisticsSnapshot else evidence.evidence_id)
        for name, value in (('status', status), ('evidence_status', evidence_status),
                            ('reasons', tuple(sorted(reasons))), ('source_references', tuple(sorted(set(refs))))):
            object.__setattr__(self, name, value)
        _identity(self, 'assessment_id')


class CalibrationAssessmentBuilder:
    def build(self, issuance, *, statistics=None, outcomes=None, assumptions=(), provenance=None):
        return CalibrationAssessment(issuance, statistics, outcomes, assumptions,
                                     {} if provenance is None else provenance)
