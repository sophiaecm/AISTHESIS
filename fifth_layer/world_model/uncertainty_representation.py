"""Source-preserving uncertainty inventory; no inference, pooling or feedback."""
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
import json
from math import fsum, isclose

from ._structured import freeze, identifier, number
from .common_evidence_state import CommonEvidenceState, bounded_plain
from .evidence import stable_id
from .physical_state import PhysicalWorldState
from .physics_constraints import PhysicsConstraintBundle
from .latent_physical_state import LatentPhysicalState
from .cross_modal_consequences import CrossModalConsequenceBundle
from .multiple_futures import MultipleFutureBundle
from .bayesian_belief_state import BayesianBeliefState
from .prediction_records import PredictionRecord
from .spatial_grounding import GroundedObservationTarget
from .active_perception import ObservationRequestPlan
from ..confidence_calibration import MIN_CALIBRATION_SAMPLES

VERSION = 'uncertainty-representation-v0.1'
CATEGORIES = ('observational', 'spatial', 'temporal', 'physical', 'cross_modal',
              'hypothesis', 'belief', 'prediction', 'calibration', 'structural')
STATUSES = ('available', 'unavailable', 'indeterminate', 'insufficient_evidence', 'not_applicable')
VALUE_KINDS = ('numeric', 'categorical', 'structured', 'confidence',
               'scalar_uncertainty', 'branch_label_distribution', 'backend_score',
               'calibration_reliability')


def _strings(value):
    if not isinstance(value, (tuple, list)):
        raise ValueError('ordered string sequence required')
    for item in value:
        identifier(item, 'reference')
    return tuple(sorted(set(value)))


def _metadata(value):
    if not isinstance(value, Mapping):
        raise ValueError('structured mapping required')
    return freeze(bounded_plain(value))


class _Record:
    def to_dict(self):
        return bounded_plain(self)

    def to_json(self):
        return json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'), allow_nan=False)


@dataclass(frozen=True)
class UncertaintyContext(_Record):
    scene_id: str
    timestamp: float | None = None
    session_id: str | None = None
    coordinate_frame_id: str | None = None

    def __post_init__(self):
        for name in ('scene_id', 'session_id', 'coordinate_frame_id'):
            identifier(getattr(self, name), name, optional=name != 'scene_id')
        if self.timestamp is not None:
            number(self.timestamp, 'timestamp', nonnegative=True)
        bounded_plain(self)


def _aligned(source, authority):
    """Known identities must agree; source times may precede the bundle cutoff."""
    for name in ('scene_id', 'session_id', 'coordinate_frame_id'):
        a, b = getattr(source, name), getattr(authority, name)
        if a is not None and b is not None and a != b:
            raise ValueError(name + ' mismatch')
    if source.timestamp is not None and (
            authority.timestamp is None or source.timestamp > authority.timestamp):
        raise ValueError('timestamp exceeds or cannot be ordered against bundle cutoff')


@dataclass(frozen=True)
class UncertaintyEntry(_Record):
    context: UncertaintyContext
    source_layer: str
    source_type: str
    source_id: str
    uncertainty_kind: str
    semantic_scope: str
    status: str
    value: object
    value_kind: str
    original_field: str
    source_schema: str | None = None
    unit: str | None = None
    assumptions: tuple = ()
    source_references: tuple = ()
    provenance: Mapping = field(default_factory=dict)
    transformation: str = 'none'
    uncertainty_id: str = field(default='', init=False)
    schema_version: str = field(default=VERSION, init=False)
    rule_version: str = field(default=VERSION, init=False)

    def __post_init__(self):
        if not isinstance(self.context, UncertaintyContext):
            raise ValueError('UncertaintyContext required')
        for name in ('source_layer', 'source_type', 'source_id', 'semantic_scope', 'original_field'):
            identifier(getattr(self, name), name)
        for name in ('source_schema', 'unit'):
            identifier(getattr(self, name), name, optional=True)
        if self.uncertainty_kind not in CATEGORIES or self.status not in STATUSES or self.value_kind not in VALUE_KINDS:
            raise ValueError('unsupported category, status or value kind')
        if self.transformation != 'none':
            raise ValueError('v0.1 supports copied values only')
        if self.status == 'available' and self.value is None:
            raise ValueError('available requires an explicit value')
        if self.status in ('unavailable', 'not_applicable') and self.value is not None:
            raise ValueError('unavailable/not_applicable requires None; retain diagnostics in provenance')
        value = freeze(bounded_plain(self.value))
        if value is not None:
            if self.value_kind in ('numeric', 'confidence', 'scalar_uncertainty', 'backend_score', 'calibration_reliability'):
                number(value, self.value_kind, unit=self.value_kind in (
                    'confidence', 'scalar_uncertainty', 'calibration_reliability'))
            elif self.value_kind == 'branch_label_distribution':
                if not isinstance(value, Mapping) or not value:
                    raise ValueError('nonempty categorical branch-label distribution required')
                for label, probability in value.items():
                    identifier(label, 'branch label')
                    number(probability, 'posterior_probability', unit=True)
                if not isclose(fsum(value.values()), 1., abs_tol=1e-12, rel_tol=0.):
                    raise ValueError('branch-label distribution must sum to one')
            elif self.value_kind == 'categorical' and not isinstance(value, (str, tuple)):
                raise ValueError('categorical value requires strings')
            if self.value_kind == 'categorical':
                for item in (value,) if isinstance(value, str) else value:
                    identifier(item, 'categorical value')
        object.__setattr__(self, 'value', value)
        object.__setattr__(self, 'assumptions', _strings(self.assumptions))
        object.__setattr__(self, 'source_references', _strings(self.source_references))
        object.__setattr__(self, 'provenance', _metadata(self.provenance))
        data = self.to_dict()
        data.pop('uncertainty_id')
        object.__setattr__(self, 'uncertainty_id', stable_id('uncertainty', VERSION, data))


@dataclass(frozen=True)
class UncertaintyBundle(_Record):
    context: UncertaintyContext
    entries: tuple[UncertaintyEntry, ...]
    provenance: Mapping = field(default_factory=dict)
    source_lineage: tuple = field(default=(), init=False)
    bundle_id: str = field(default='', init=False)
    schema_version: str = field(default=VERSION, init=False)
    rule_version: str = field(default=VERSION, init=False)

    def __post_init__(self):
        if not isinstance(self.context, UncertaintyContext):
            raise ValueError('UncertaintyContext required')
        if not isinstance(self.entries, (tuple, list)) or any(not isinstance(e, UncertaintyEntry) for e in self.entries):
            raise ValueError('typed entries required')
        entries = tuple(sorted(self.entries, key=lambda e: e.uncertainty_id))
        if len({e.uncertainty_id for e in entries}) != len(entries):
            raise ValueError('duplicate uncertainty entry')
        for e in entries:
            _aligned(e.context, self.context)
        # An unspecified bundle field must not hide conflicting declared sources.
        for name in ('session_id', 'coordinate_frame_id'):
            if len({getattr(e.context, name) for e in entries if getattr(e.context, name) is not None}) > 1:
                raise ValueError(name + ' mismatch between sources')
        object.__setattr__(self, 'entries', entries)
        lineage = sorted({(e.source_layer, e.source_type, e.source_id) for e in entries})
        object.__setattr__(self, 'source_lineage', tuple(lineage))
        object.__setattr__(self, 'provenance', _metadata(self.provenance))
        data = self.to_dict()
        data.pop('bundle_id')
        object.__setattr__(self, 'bundle_id', stable_id('uncertainty-bundle', VERSION, data))


def _typed(source, cls, schema=None):
    if type(source) is not cls or (schema is not None and source.schema_version != schema):
        raise ValueError('expected ' + cls.__name__ + ' with supported schema')


def _context(source, time_field='timestamp'):
    provenance = getattr(source, 'provenance', {})
    context = UncertaintyContext(source.scene_id, getattr(source, time_field),
        getattr(source, 'session_id', provenance.get('session_id')),
        getattr(source, 'coordinate_frame_id', provenance.get('coordinate_frame_id')))
    for name in ('scene_id', 'session_id', 'coordinate_frame_id'):
        declared = provenance.get(name)
        if declared is not None and getattr(context, name) is not None and declared != getattr(context, name):
            raise ValueError('source provenance ' + name + ' mismatch')
    return context


def _entry(source, context, source_id, category, path, value, *, status=None,
           kind='structured', scope=None, refs=(), assumptions=(), details=None, unit=None):
    return UncertaintyEntry(context, type(source).__module__, type(source).__name__, source_id,
        category, scope or path, status or ('unavailable' if value is None else 'available'), value,
        kind, path, getattr(source, 'schema_version', None), unit, assumptions, refs,
        dict(source_provenance=getattr(source, 'provenance', {}), source_details=details or {}))


def _status(value):
    # This describes availability/ambiguity of the source field, never world truth.
    return {'unavailable': 'unavailable', 'unsupported': 'unavailable',
            'unknown': 'indeterminate', 'indeterminate': 'indeterminate',
            'not_applicable': 'not_applicable', 'invalid_evidence': 'unavailable'}.get(value, 'available')


def adapt_physical_uncertainty(source):
    _typed(source, PhysicalWorldState, 'physical-world-0.1')
    return (_entry(source, _context(source), source.scene_id, 'physical', 'uncertainty',
        source.uncertainty or None, refs=source.evidence_references,
        details={'source_identity_kind': 'scene_snapshot', 'objects': source.objects, 'relations': source.relations}),)


def adapt_constraint_uncertainty(source):
    _typed(source, PhysicsConstraintBundle, 'physics-constraints-0.2')
    context = _context(source)
    if not source.results:
        return (_entry(source, context, source.scene_id, 'physical', 'results', None,
            details={'source_identity_kind': 'scene_snapshot', 'empty_results_not_certainty': True}),)
    return tuple(_entry(r, context, r.constraint_id, 'physical', 'uncertainty',
        None if _status(r.status) in ('unavailable', 'not_applicable') else r.uncertainty,
        status=_status(r.status), refs=r.derived_from,
        details={'status': r.status, 'finding': r.finding, 'uncertainty': r.uncertainty,
                 'source_bundle_schema': source.schema_version, 'rule_version': r.rule_version}) for r in source.results)


def adapt_latent_uncertainty(source):
    _typed(source, LatentPhysicalState, 'latent-physical-state-0.1')
    context = _context(source)
    entries = [_entry(source, context, source.latent_state_id, 'physical', 'uncertainty', source.uncertainty or None)]
    for name in ('objects', 'relations', 'dynamics', 'active_constraints'):
        for record in getattr(source, name):
            # A content reference is explicitly not an invented upstream object ID.
            ref = stable_id('latent-content-reference', source.latent_state_id, name, bounded_plain(record))
            status = _status(record.get('status'))
            value = record.get('uncertainty')
            entries.append(_entry(source, context, source.latent_state_id, 'physical', name + '.uncertainty',
                None if status in ('unavailable', 'not_applicable') else value,
                status=status if status != 'available' else None, refs=(ref,),
                details={'source_reference_kind': 'content_reference', 'source_record': record}))
    return tuple(entries)


def adapt_cross_modal_uncertainty(source):
    _typed(source, CrossModalConsequenceBundle, 'cross-modal-consequences-0.2')
    context = _context(source)
    ref = stable_id('consequence-bundle-reference', source.to_dict())
    entries = [_entry(source, context, ref, 'cross_modal', 'uncertainty', source.uncertainty or None,
                      details={'source_identity_kind': 'content_reference'})]
    for c in source.candidates:
        status = _status(c.status)
        entries.append(_entry(c, context, c.consequence_id, 'cross_modal', 'uncertainty',
            None if status == 'unavailable' else c.uncertainty, status=status,
            refs=c.source_field_references, details={'status': c.status, 'modality': c.modality,
                'event_family': c.event_family, 'uncertainty': c.uncertainty}))
    return tuple(entries)


def adapt_future_uncertainty(source):
    _typed(source, MultipleFutureBundle, 'multiple-futures-0.2')
    context = _context(source)
    ref = stable_id('future-bundle-reference', source.to_dict())
    entries = [_entry(source, context, ref, 'hypothesis', 'branch_relations', source.branch_relations,
        refs=tuple(b.future_id for b in source.branches), details={'uncertainty': source.uncertainty,
        'source_identity_kind': 'content_reference', 'empty_branches_not_impossibility': True})]
    for b in source.branches:
        status = _status(b.status)
        entries.append(_entry(b, context, b.future_id, 'hypothesis', 'uncertainty',
            None if status == 'unavailable' else b.uncertainty, status=status,
            assumptions=tuple(a.assumption_id for a in b.assumptions), refs=b.source_field_references,
            details={'status': b.status, 'assumptions': b.assumptions, 'uncertainty': b.uncertainty}))
        entries.append(_entry(b, context, b.future_id, 'temporal', 'horizon_value', b.horizon_value,
            details={'horizon_kind': b.horizon_kind, 'qualitative_horizon_not_duration': True}))
    return tuple(entries)


def adapt_belief_uncertainty(source):
    _typed(source, BayesianBeliefState, 'bayesian-belief-state-v0.1')
    probabilities = {b.hypothesis_id: b.posterior_probability for b in source.beliefs}
    present = [v is not None for v in probabilities.values()]
    if any(present) and not all(present):
        raise ValueError('partial posterior distribution unsupported')
    value = probabilities if present and all(present) else None
    status = _status(source.update_status) if value is None else 'available'
    if status == 'available' and value is None:
        status = 'unavailable'
    return (_entry(source, _context(source), source.belief_state_id, 'belief',
        'beliefs.posterior_probability', value, status=status, kind='branch_label_distribution',
        scope=source.probability_scope, refs=source.source_future_ids,
        assumptions=tuple(a for b in source.beliefs for a in b.assumptions),
        details={'update_status': source.update_status, 'uncertainty': source.uncertainty,
                 'beliefs': source.beliefs, 'evidence_lineage': source.evidence_lineage}),)


def adapt_prediction_uncertainty(source):
    _typed(source, PredictionRecord)
    return tuple(_entry(source, _context(source, 'source_timestamp'), source.prediction_id, 'prediction',
        name, getattr(source, name), kind=kind, refs=(*source.evidence_for, *source.evidence_against),
        details={'hypothesis_id': source.hypothesis_id, 'target_timestamp': source.target_timestamp,
                 'horizon_seconds': source.horizon_seconds, 'prediction_not_observation': True})
        for name, kind in (('confidence', 'confidence'), ('uncertainty', 'scalar_uncertainty')))


def adapt_calibration_uncertainty(source, *, context, source_id):
    """Exact output of ConfidenceCalibrationMemory.calibrate; never calls memory."""
    keys = {'raw_confidence', 'calibration_reliability', 'calibrated_confidence',
            'calibration_samples', 'calibration_bucket'}
    if not isinstance(source, Mapping) or set(source) != keys:
        raise ValueError('exact calibrate output required')
    copied = bounded_plain(source)
    for name in ('raw_confidence', 'calibration_reliability', 'calibrated_confidence'):
        number(copied[name], name, unit=True)
    n, bucket = copied['calibration_samples'], copied['calibration_bucket']
    if type(n) is not int or n < 0 or not isinstance(bucket, list) or not 1 <= len(bucket) <= 3:
        raise ValueError('invalid calibration count or bucket')
    for i, part in enumerate(bucket):
        identifier(part, 'bucket', optional=i == 2)
    prior = bucket == ['global_prior']
    if prior != (n < MIN_CALIBRATION_SAMPLES):
        raise ValueError('calibration bucket/count mismatch')
    status = 'insufficient_evidence' if prior else 'available'
    return (UncertaintyEntry(context, 'fifth_layer.confidence_calibration',
        'ConfidenceCalibrationMemory.calibrate', source_id, 'calibration',
        'bounded_session_finalized_evaluable_outcome_reliability', status,
        copied['calibration_reliability'], 'calibration_reliability', 'calibration_reliability',
        provenance={'source_output': copied, 'context_origin': 'caller_supplied',
                    'minimum_samples': MIN_CALIBRATION_SAMPLES,
                    'reliability_not_physical_truth': True}),)


def adapt_grounding_uncertainty(source):
    _typed(source, GroundedObservationTarget, 'spatial-grounding-v0.1')
    context = _context(source.frame)
    status = {'no_candidate': 'unavailable', 'multiple_candidates': 'indeterminate',
              'grounded_candidate': 'available', 'unavailable': 'unavailable',
              'indeterminate': 'indeterminate'}[source.grounding_status]
    entries = [_entry(source, context, source.grounded_target_id, 'spatial', 'uncertainty',
        None if status == 'unavailable' else source.uncertainty, status=status,
        refs=(source.source_request_id, source.frame.frame_id),
        details={'grounding_status': source.grounding_status, 'uncertainty': source.uncertainty,
                 'backend_metadata': source.backend_metadata, 'regions': source.regions,
                 'no_candidate_not_physical_absence': True})]
    for metadata in source.backend_metadata:
        entries.append(_entry(source, context, metadata['source_observation_id'], 'spatial',
            'backend_metadata.backend_score', metadata.get('backend_score'), kind='backend_score',
            refs=(source.grounded_target_id,), scope='backend_specific_score_not_confidence_or_probability',
            details=metadata))
    return tuple(entries)


def adapt_active_perception_uncertainty(source):
    _typed(source, ObservationRequestPlan, 'active-perception-v0.1')
    context = _context(source)
    entries = [_entry(source, context, source.plan_id, 'structural', 'uncertainty', source.uncertainty or None)]
    for cue in source.cues:
        entries.append(_entry(cue, context, cue.cue_id, 'structural', 'uncertainty', cue.uncertainty,
            status='indeterminate' if cue.epistemic_status == 'indeterminate' else 'available',
            refs=cue.source_ids, details={'cue_type': cue.cue_type, 'epistemic_status': cue.epistemic_status}))
    return tuple(entries)


def adapt_common_evidence_uncertainty(source):
    _typed(source, CommonEvidenceState, 'common-evidence-state-0.1')
    context = _context(source)
    entries = []
    for family, status in source.availability.items():
        entries.append(_entry(source, context, source.scene_id, 'structural', 'availability.' + family,
            None if status == 'unavailable' else status, status=status, kind='categorical'))
    for item in source.evidence_items:
        item_context = _context(item)
        category = 'observational' if item.epistemic_status == 'observed' else 'structural'
        for path, value in source.uncertainty['sources'][item.evidence_id].items():
            entries.append(_entry(item, item_context, item.evidence_id, category, path, value,
                details={'epistemic_status': item.epistemic_status, 'source_family': item.source_type}))
        entries.append(_entry(item, item_context, item.evidence_id, category, 'confidence',
            item.confidence, kind='confidence', details={'epistemic_status': item.epistemic_status}))
    entries.append(_entry(source, context, source.scene_id, 'structural', 'uncertainty.integration',
                          source.uncertainty['integration']))
    return tuple(entries)


def adapt_reasoning_uncertainty(source):
    # Keep orchestration imports off the core record import path.
    from ..reasoning_connectome_v02 import ReasoningGraphSnapshotV02
    _typed(source, ReasoningGraphSnapshotV02, 'reasoning-connectome-v0.2')
    return (_entry(source, _context(source), source.snapshot_id, 'structural', 'uncertainty',
        source.uncertainty, refs=(source.context_reference,),
        details={'source_references': source.source_references,
                 'layer_availability': source.layer_availability,
                 'routing_relevance_not_truth_probability': True}),)


class UncertaintyRepresentationBuilder:
    def build(self, context, entries, *, provenance=None):
        return UncertaintyBundle(context, entries, {} if provenance is None else provenance)


def bundle_from_dict(data):
    """Strict round trip of this schema only; verifies content IDs and lineage."""
    if not isinstance(data, Mapping):
        raise ValueError('bundle mapping required')
    plain = bounded_plain(data)
    try:
        def restore(cls, value):
            expected = {f.name for f in fields(cls)}
            if set(value) != expected:
                raise ValueError('unexpected or missing serialized fields')
            kwargs = {f.name: value[f.name] for f in fields(cls) if f.init}
            if 'context' in kwargs:
                kwargs['context'] = UncertaintyContext(**kwargs['context'])
            if cls is UncertaintyBundle:
                kwargs['entries'] = tuple(restore(UncertaintyEntry, e) for e in kwargs['entries'])
            result = cls(**kwargs)
            if result.to_dict() != value:
                raise ValueError('serialized identity, ordering or version mismatch')
            return result
        return restore(UncertaintyBundle, plain)
    except (KeyError, TypeError) as exc:
        raise ValueError('invalid serialized uncertainty bundle') from exc
