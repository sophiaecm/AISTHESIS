"""Descriptive metrics gated by source semantics; no inference or calibration."""
from collections.abc import Mapping
from dataclasses import dataclass, field
import json

from ._structured import freeze, geometry, identifier, number
from .common_evidence_state import bounded_plain
from .evidence import stable_id
from .calibration_assessment import CalibrationIssuance, FinalizedCalibrationOutcome, adapt_live_prediction
from ..confidence_calibration import OUTCOME_WEIGHTS
from ..evaluation_contracts import EVALUATION_GRACE_SECONDS

VERSION = 'evaluation-metrics-v0.1'
WEIGHTS = freeze(dict(OUTCOME_WEIGHTS))
FINAL_STATUSES = ('correct', 'partially_correct', 'incorrect', 'unevaluable', 'expired')
METRICS = ('evaluation_coverage', 'correct_outcome_frequency', 'partial_outcome_frequency',
           'incorrect_outcome_frequency', 'weighted_outcome_score', 'strict_accuracy',
           'incorrect_outcome_rate', 'brier_score', 'nll', 'ece', 'tta')
PROBABILITY_METRICS = ('brier_score', 'nll', 'ece')
SOURCE_SEMANTICS = ('engineering_confidence_and_geometric_outcomes',
                    'bayesian_branch_label_distribution', 'multiple_future_possibilities')
_RESULT_FIELDS = frozenset(('prediction_id', 'track_id', 'prediction_created_timestamp',
    'prediction_horizon_seconds', 'target_timestamp', 'source', 'status', 'center_error_pixels',
    'normalized_center_error', 'bbox_iou', 'direction_match', 'timing_error_seconds',
    'evaluated_timestamp', 'evaluation_reason', 'prediction', 'observation'))
_MEASUREMENTS = ('center_error_pixels', 'normalized_center_error', 'bbox_iou',
                 'direction_match', 'timing_error_seconds')


def _mapping(value):
    if not isinstance(value, Mapping):
        raise ValueError('structured mapping required')
    return freeze(bounded_plain(value))


def _strings(values):
    if not isinstance(values, (tuple, list)):
        raise ValueError('ordered references required')
    for value in values:
        identifier(value, 'reference')
    return tuple(sorted(set(values)))


def _count(value):
    if type(value) is not int or value < 0:
        raise ValueError('nonnegative integer count required')
    number(value, 'count')


class _Record:
    def to_dict(self):
        return bounded_plain(self)

    def to_json(self):
        return json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'), allow_nan=False)


def _identity(record, name):
    data = record.to_dict()
    data.pop(name)
    object.__setattr__(record, name, stable_id(name, VERSION, data))


def _finalized_result(source, issuance):
    data = _mapping(source)
    if set(data) != _RESULT_FIELDS or data['status'] not in FINAL_STATUSES:
        raise ValueError('current finalized prediction evaluation contract required')
    # Reuse public issuance validation; never execute prediction or calibration.
    embedded = adapt_live_prediction(data['prediction'], context=issuance.context, source_reference='embedded_prediction')
    for name in ('prediction_id', 'prediction_source', 'prediction_type', 'class_name', 'calibration'):
        if getattr(embedded, name) != getattr(issuance, name):
            raise ValueError('outcome differs from issued prediction: ' + name)
    p = data['prediction']
    original = issuance.provenance.get('source_prediction')
    if original is not None and original != p:
        raise ValueError('issued prediction content was rewritten')
    if issuance.prediction_created_timestamp is not None and issuance.prediction_created_timestamp != p['prediction_created_timestamp']:
        raise ValueError('prediction origin mismatch')
    for name in ('prediction_id', 'track_id', 'prediction_created_timestamp',
                 'prediction_horizon_seconds', 'target_timestamp', 'source'):
        if data[name] != p[name]:
            raise ValueError('inconsistent result lineage: ' + name)
    number(data['evaluated_timestamp'], 'evaluated_timestamp', nonnegative=True)
    if data['evaluated_timestamp'] < p['target_timestamp'] + EVALUATION_GRACE_SECONDS:
        raise ValueError('evaluation precedes closure of the producer finalization window')
    issued = issuance.context.timestamp
    if issued is not None and issued > data['evaluated_timestamp']:
        raise ValueError('evaluation precedes issuance')
    for name in _MEASUREMENTS:
        value = data[name]
        if value is not None:
            if name == 'direction_match':
                if type(value) is not bool:
                    raise ValueError('direction_match requires boolean or None')
            else:
                number(value, name, unit=name == 'bbox_iou', nonnegative=name != 'timing_error_seconds')
    status = data['status']
    if status in WEIGHTS:
        if data['evaluation_reason'] != 'same_track_observed_geometry' or any(data[n] is None for n in _MEASUREMENTS):
            raise ValueError('evaluable source requires evaluation measurements and reason')
        observation = data['observation']
        if not isinstance(observation, Mapping) or observation.get('observation_state') != 'observed' or observation.get('is_predicted'):
            raise ValueError('evaluable reference must be an observation')
        if observation.get('track_id') != p['track_id']:
            raise ValueError('reference track mismatch')
        for name, size in (('observed_center', 2), ('observed_bbox', 4)):
            geometry(observation.get(name), name, size)
        for name, size in (('origin_center', 2), ('predicted_center', 2), ('predicted_bbox', 4), ('image_size', 2)):
            geometry(p[name], name, size)
        if any(v <= 0 for v in p['image_size']):
            raise ValueError('positive image dimensions required')
        stamp = observation.get('observed_timestamp')
        number(stamp, 'observed_timestamp', nonnegative=True)
        if not p['prediction_created_timestamp'] < stamp <= data['evaluated_timestamp']:
            raise ValueError('invalid reference chronology')
        if abs(stamp - p['target_timestamp']) > EVALUATION_GRACE_SECONDS:
            raise ValueError('reference does not match the forecast target window')
        if issued is not None and issued >= stamp:
            raise ValueError('forecast must be issued strictly before reference observation')
        if observation.get('image_size', p['image_size']) != p['image_size']:
            raise ValueError('evaluable reference geometry differs from prediction')
    elif status == 'unevaluable':
        if data['evaluation_reason'] not in ('no_suitable_observation', 'image_geometry_changed') or any(data[n] is not None for n in _MEASUREMENTS):
            raise ValueError('invalid unevaluable source measurements or reason')
    elif data['evaluation_reason'] != 'memory_limit_or_ttl':
        raise ValueError('expiration requires source expiration reason')
    return data


@dataclass(frozen=True)
class EvaluationUnit(_Record):
    issuance: CalibrationIssuance
    lifecycle: str
    source_reference: str
    outcome: Mapping | None = None
    reference_id: str | None = None
    provenance: Mapping = field(default_factory=dict)
    outcome_eligible: bool = field(default=False, init=False)
    exclusion_reasons: tuple = field(default=(), init=False)
    unit_id: str = field(default='', init=False)
    schema_version: str = field(default=VERSION, init=False)
    source_semantics: str = field(default=SOURCE_SEMANTICS[0], init=False)
    reference_semantics: str = field(default='repository_observed_track_geometry_not_physical_truth', init=False)

    def __post_init__(self):
        if type(self.issuance) is not CalibrationIssuance:
            raise ValueError('explicit CalibrationIssuance required')
        if self.lifecycle not in ('finalized', 'pending', 'missing_outcome'):
            raise ValueError('unsupported evaluation lifecycle')
        identifier(self.source_reference, 'source_reference')
        identifier(self.reference_id, 'reference_id', optional=True)
        reasons = []
        if self.lifecycle != 'finalized':
            if self.outcome is not None or self.reference_id is not None:
                raise ValueError('unresolved prediction cannot contain a finalized reference')
            reasons.append(self.lifecycle)
        else:
            data = _finalized_result(self.outcome, self.issuance)
            object.__setattr__(self, 'outcome', data)
            if data['status'] not in WEIGHTS:
                reasons.append(data['status'])
            else:
                if self.reference_id is None:
                    reasons.append('reference_identity_unavailable')
                if self.issuance.context.timestamp is None:
                    reasons.append('issuance_time_unavailable')
        object.__setattr__(self, 'provenance', _mapping(self.provenance))
        object.__setattr__(self, 'exclusion_reasons', tuple(sorted(reasons)))
        object.__setattr__(self, 'outcome_eligible', not reasons)
        _identity(self, 'unit_id')


def adapt_finalized_evaluation(source, *, issuance, source_reference, reference_id=None,
                               finalized, provenance=None):
    """Caller attests finalized history output; no outcome is generated here."""
    if finalized is not True:
        raise ValueError('explicit finalization declaration required')
    if type(source) is FinalizedCalibrationOutcome:
        source = source.result
    return EvaluationUnit(issuance, 'finalized', source_reference, source, reference_id,
        {'finalization': 'caller_attested', 'caller_provenance': {} if provenance is None else provenance})


def represent_unresolved_prediction(issuance, *, lifecycle, source_reference, provenance=None):
    """Explicit pending inventory or missing evaluation; never an outcome label."""
    if lifecycle not in ('pending', 'missing_outcome'):
        raise ValueError('unresolved lifecycle required')
    return EvaluationUnit(issuance, lifecycle, source_reference, provenance={} if provenance is None else provenance)


@dataclass(frozen=True)
class MetricEligibility(_Record):
    metric_name: str
    available_sample_count: int
    source_semantics: str = SOURCE_SEMANTICS[0]
    status: str = field(default='', init=False)
    eligible: bool = field(default=False, init=False)
    reasons: tuple = field(default=(), init=False)
    requirements_met: tuple = field(default=(), init=False)
    requirements_missing: tuple = field(default=(), init=False)
    target_semantics: str = field(default='repository_geometric_outcome_labels', init=False)
    policy_version: str = field(default=VERSION, init=False)

    def __post_init__(self):
        if self.metric_name not in METRICS or self.source_semantics not in SOURCE_SEMANTICS:
            raise ValueError('unsupported metric or source semantics')
        _count(self.available_sample_count)
        if self.source_semantics != SOURCE_SEMANTICS[0]:
            object.__setattr__(self, 'target_semantics', 'no_resolved_event_target_supplied')
        met, missing = (), ()
        if self.metric_name in PROBABILITY_METRICS:
            status = 'ineligible'
            missing = ('contractual_probability_forecast', 'resolved_same_event_probability_target')
            reasons = ('bounded_or_normalized_value_not_probability_forecast',)
            if self.source_semantics != SOURCE_SEMANTICS[0]:
                missing += ('mutually_exclusive_exhaustive_event_semantics',)
                reasons += ('branch_labels_not_ground_truth_event_classes',)
        elif self.metric_name == 'tta':
            status = 'protocol_required'
            missing = ('event_onset', 'valid_anticipation', 'anticipation_threshold',
                       'temporal_alignment_policy', 'false_anticipation_policy', 'censoring_policy')
            reasons = ('experiment_001_tta_definition_not_fixed',)
        elif self.source_semantics != SOURCE_SEMANTICS[0]:
            status = 'ineligible'
            missing = ('supported_finalized_outcome_contract',)
            reasons = ('no_outcome_label_conversion_for_branches',)
        else:
            met = ('explicit_denominator_policy',)
            if self.metric_name == 'evaluation_coverage':
                met += ('explicit_supplied_prediction_inventory',)
                required = 'nonempty_supplied_inventory'
            else:
                met += ('finalized_evaluable_source', 'issued_before_reference', 'explicit_observation_reference')
                required = 'nonempty_eligible_outcome_set'
            if self.available_sample_count:
                status, reasons = 'eligible', ('descriptive_only_not_scientific_validation',)
                met += (required,)
            else:
                status, reasons, missing = 'insufficient_data', ('empty_denominator',), (required,)
            if self.metric_name == 'strict_accuracy':
                met += ('only_source_correct_status_counts_as_success',)
        for name, value in (('status', status), ('eligible', status == 'eligible'), ('reasons', tuple(sorted(reasons))),
                            ('requirements_met', tuple(sorted(met))), ('requirements_missing', tuple(sorted(missing)))):
            object.__setattr__(self, name, value)
        bounded_plain(self)


def probability_metric_eligibility(metric_name, source):
    """Explicitly explain ineligibility of existing probability-like sources."""
    from .bayesian_belief_state import BayesianBeliefState
    from .multiple_futures import MultipleFutureBundle
    if metric_name not in PROBABILITY_METRICS:
        raise ValueError('probability metric name required')
    kinds = {CalibrationIssuance: SOURCE_SEMANTICS[0], BayesianBeliefState: SOURCE_SEMANTICS[1],
             MultipleFutureBundle: SOURCE_SEMANTICS[2]}
    if type(source) not in kinds:
        raise ValueError('known typed source required; numeric scores are not probability contracts')
    return MetricEligibility(metric_name, 0, kinds[type(source)])


@dataclass(frozen=True)
class MetricResult(_Record):
    eligibility: MetricEligibility
    numerator: float | None
    denominator: int | None
    denominator_definition: str
    contributing_unit_ids: tuple
    source_references: tuple
    interpretation: str
    value: float | None = field(default=None, init=False)
    sample_count: int = field(default=0, init=False)
    metric_name: str = field(default='', init=False)
    transformation: str = field(default='', init=False)
    policy_version: str = field(default=VERSION, init=False)

    def __post_init__(self):
        if type(self.eligibility) is not MetricEligibility:
            raise ValueError('MetricEligibility required')
        for name in ('denominator_definition', 'interpretation'):
            identifier(getattr(self, name), name)
        ids = _strings(self.contributing_unit_ids)
        if len(ids) != len(self.contributing_unit_ids):
            raise ValueError('duplicate contributing unit')
        refs = _strings(self.source_references)
        object.__setattr__(self, 'contributing_unit_ids', ids)
        object.__setattr__(self, 'source_references', refs)
        object.__setattr__(self, 'metric_name', self.eligibility.metric_name)
        if self.eligibility.eligible:
            number(self.numerator, 'numerator', nonnegative=True)
            _count(self.denominator)
            if not self.denominator or self.numerator > self.denominator or self.denominator != len(ids):
                raise ValueError('invalid metric numerator or denominator')
            if self.eligibility.available_sample_count != self.denominator or not refs:
                raise ValueError('metric evidence count or references inconsistent')
            object.__setattr__(self, 'value', self.numerator / self.denominator)
            object.__setattr__(self, 'sample_count', self.denominator)
            object.__setattr__(self, 'transformation', 'explicit_numerator_divided_by_denominator')
        else:
            if self.numerator is not None or self.denominator is not None or ids:
                raise ValueError('ineligible/unavailable metric cannot contain numeric placeholders or contributors')
            object.__setattr__(self, 'transformation', 'not_computed')
        bounded_plain(self)


@dataclass(frozen=True)
class OutcomeDistribution(_Record):
    correct_count: int
    partial_count: int
    incorrect_count: int
    evaluable_count: int

    def __post_init__(self):
        for value in (self.correct_count, self.partial_count, self.incorrect_count, self.evaluable_count):
            _count(value)
        if self.evaluable_count != self.correct_count + self.partial_count + self.incorrect_count:
            raise ValueError('inconsistent outcome counts')


@dataclass(frozen=True)
class EvaluationCoverage(_Record):
    total_records: int
    evaluable_count: int
    excluded_count: int
    source_evaluable_count: int
    unqualified_evaluable_count: int
    unevaluable_count: int
    expired_count: int
    pending_count: int
    missing_outcome_count: int

    def __post_init__(self):
        for value in self.to_dict().values():
            _count(value)
        if (self.total_records != self.evaluable_count + self.excluded_count
                or self.source_evaluable_count != self.evaluable_count + self.unqualified_evaluable_count
                or self.total_records != self.source_evaluable_count + self.unevaluable_count + self.expired_count
                   + self.pending_count + self.missing_outcome_count):
            raise ValueError('inconsistent coverage accounting')


def _references(units):
    return tuple(sorted({ref for u in units for ref in
        (u.unit_id, u.source_reference, u.issuance.source_reference, u.reference_id) if ref is not None}))


def _results(units, distribution):
    eligible = tuple(u for u in units if u.outcome_eligible)
    n = distribution.evaluable_count
    numerators = dict(evaluation_coverage=n, correct_outcome_frequency=distribution.correct_count,
        partial_outcome_frequency=distribution.partial_count, incorrect_outcome_frequency=distribution.incorrect_count,
        strict_accuracy=distribution.correct_count, incorrect_outcome_rate=distribution.incorrect_count,
        weighted_outcome_score=(WEIGHTS['correct'] * distribution.correct_count +
                               WEIGHTS['partially_correct'] * distribution.partial_count +
                               WEIGHTS['incorrect'] * distribution.incorrect_count))
    results = []
    for name in METRICS:
        coverage = name == 'evaluation_coverage'
        selected = units if coverage else eligible
        eligibility = MetricEligibility(name, len(selected))
        definition = 'all_supplied_unique_prediction_units' if coverage else 'finalized_evaluable_units_with_reference_and_issuance_chronology'
        interpretation = {'weighted_outcome_score': 'weighted outcome success using original 1/0.5/0 weights; not accuracy or probability',
            'strict_accuracy': 'only original correct status is success; partial and incorrect are not strict success',
            'incorrect_outcome_rate': 'direct incorrect count; not one minus weighted success',
            'evaluation_coverage': 'fraction of supplied inventory eligible for descriptive outcome measurement; not accuracy'}.get(
                name, 'source outcome frequency, not physical truth probability')
        if not eligibility.eligible:
            interpretation = '; '.join(eligibility.reasons)
        results.append(MetricResult(eligibility, numerators[name] if eligibility.eligible else None,
            len(selected) if eligibility.eligible else None, definition,
            tuple(u.unit_id for u in selected) if eligibility.eligible else (),
            _references(selected if eligibility.eligible else units), interpretation))
    return tuple(sorted(results, key=lambda r: r.metric_name))


@dataclass(frozen=True)
class EvaluationMetricsReport(_Record):
    dataset_id: str
    units: tuple[EvaluationUnit, ...]
    as_of_timestamp: float | None = None
    session_id: str | None = None
    assumptions: tuple = ()
    provenance: Mapping = field(default_factory=dict)
    coverage: EvaluationCoverage = field(init=False)
    outcome_distribution: OutcomeDistribution = field(init=False)
    metrics: tuple[MetricResult, ...] = field(init=False)
    source_references: tuple = field(default=(), init=False)
    report_id: str = field(default='', init=False)
    schema_version: str = field(default=VERSION, init=False)
    rule_version: str = field(default=VERSION, init=False)

    def __post_init__(self):
        identifier(self.dataset_id, 'dataset_id')
        identifier(self.session_id, 'session_id', optional=True)
        if self.as_of_timestamp is not None:
            number(self.as_of_timestamp, 'as_of_timestamp', nonnegative=True)
        if not isinstance(self.units, (tuple, list)) or any(type(u) is not EvaluationUnit for u in self.units):
            raise ValueError('typed evaluation units required')
        if len({u.issuance.prediction_id for u in self.units}) != len(self.units):
            raise ValueError('duplicate prediction in supplied inventory')
        units = tuple(sorted(self.units, key=lambda u: u.unit_id))
        for u in units:
            if self.session_id is not None and u.issuance.context.session_id not in (None, self.session_id):
                raise ValueError('report session mismatch')
            times = [u.issuance.context.timestamp]
            if u.outcome is not None:
                times.append(u.outcome['evaluated_timestamp'])
            if self.as_of_timestamp is not None and any(t is not None and t > self.as_of_timestamp for t in times):
                raise ValueError('record follows report cutoff')
        object.__setattr__(self, 'units', units)
        object.__setattr__(self, 'assumptions', _strings(self.assumptions))
        object.__setattr__(self, 'provenance', _mapping(self.provenance))
        counts = {status: sum(u.outcome_eligible and u.outcome['status'] == status for u in units) for status in WEIGHTS}
        n = sum(counts.values())
        source_n = sum(u.outcome is not None and u.outcome['status'] in WEIGHTS for u in units)
        distribution = OutcomeDistribution(counts['correct'], counts['partially_correct'], counts['incorrect'], n)
        coverage = EvaluationCoverage(len(units), n, len(units) - n, source_n, source_n - n,
            sum(u.outcome is not None and u.outcome['status'] == 'unevaluable' for u in units),
            sum(u.outcome is not None and u.outcome['status'] == 'expired' for u in units),
            sum(u.lifecycle == 'pending' for u in units), sum(u.lifecycle == 'missing_outcome' for u in units))
        object.__setattr__(self, 'coverage', coverage)
        object.__setattr__(self, 'outcome_distribution', distribution)
        object.__setattr__(self, 'metrics', _results(units, distribution))
        object.__setattr__(self, 'source_references', _references(units))
        _identity(self, 'report_id')


class EvaluationMetricsBuilder:
    def build(self, dataset_id, units, *, as_of_timestamp=None, session_id=None, assumptions=(), provenance=None):
        return EvaluationMetricsReport(dataset_id, units, as_of_timestamp, session_id, assumptions,
                                       {} if provenance is None else provenance)
