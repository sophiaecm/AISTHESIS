"""Conditional consequence candidates; never sensing, event confirmation or belief."""
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
import json

from ._structured import freeze, identifier, number
from .common_evidence_state import bounded_plain
from .evidence import stable_id
from .hybrid_world_state import HybridWorldState
from .latent_physical_state import LatentPhysicalState
from .physics_constraints import PhysicsTransitionAssessment


VERSION = 'cross-modal-consequences-0.2'


class ConsequenceModality(str, Enum):
    ACOUSTIC = 'acoustic'
    TACTILE_FORCE = 'tactile_force'
    THERMAL = 'thermal'
    KINESTHETIC = 'kinesthetic'


class ConsequenceStatus(str, Enum):
    EXPECTED = 'expected'
    POSSIBLE = 'possible'
    UNAVAILABLE = 'unavailable'
    UNSUPPORTED = 'unsupported'
    INDETERMINATE = 'indeterminate'


def _strings(values, name):
    if not isinstance(values, (tuple, list)):
        raise ValueError(f'{name} must be an ordered sequence')
    for value in values:
        identifier(value, name)
    return tuple(sorted(set(values)))


def _context(record):
    for name in ('scene_id', 'session_id', 'coordinate_frame_id'):
        identifier(getattr(record, name), name, optional=name == 'coordinate_frame_id')
    if record.timestamp is not None:
        number(record.timestamp, 'timestamp', nonnegative=True)


def _metadata(value):
    if not isinstance(value, Mapping) or not value:
        raise ValueError('nonempty structured provenance required')
    return freeze(bounded_plain(value))


class _Serializable:
    def to_dict(self):
        return bounded_plain(self)

    def to_json(self):
        return json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'), allow_nan=False)


@dataclass(frozen=True)
class PhysicalConsequenceCandidate(_Serializable):
    consequence_id: str
    scene_id: str
    timestamp: float | None
    session_id: str
    coordinate_frame_id: str | None
    modality: ConsequenceModality
    event_family: str
    status: ConsequenceStatus
    object_ids: tuple[str, ...]
    source_physical_state_id: str
    source_constraint_ids: tuple[str, ...]
    source_field_references: tuple[str, ...]
    description: str
    rule_id: str
    provenance: Mapping
    uncertainty: tuple[str, ...] = ()
    confidence: float | None = None
    rule_version: str = field(default=VERSION, init=False)
    schema_version: str = field(default=VERSION, init=False)

    def __post_init__(self):
        _context(self)
        for name in ('consequence_id', 'event_family', 'source_physical_state_id', 'description', 'rule_id'):
            identifier(getattr(self, name), name)
        try:
            object.__setattr__(self, 'modality', ConsequenceModality(self.modality))
            object.__setattr__(self, 'status', ConsequenceStatus(self.status))
        except (ValueError, TypeError) as exc:
            raise ValueError('invalid consequence modality or status') from exc
        if self.confidence is not None:
            raise ValueError('v0.2 does not assign consequence probabilities')
        for name in ('object_ids', 'source_constraint_ids', 'source_field_references', 'uncertainty'):
            object.__setattr__(self, name, _strings(getattr(self, name), name))
        if not self.object_ids or not self.source_field_references:
            raise ValueError('candidate requires physical objects and source field references')
        object.__setattr__(self, 'provenance', _metadata(self.provenance))
        bounded_plain(self)


@dataclass(frozen=True)
class CrossModalConsequenceBundle(_Serializable):
    scene_id: str
    timestamp: float | None
    session_id: str
    coordinate_frame_id: str | None
    candidates: tuple[PhysicalConsequenceCandidate, ...]
    provenance: Mapping
    uncertainty: tuple[str, ...] = ()
    schema_version: str = field(default=VERSION, init=False)

    def __post_init__(self):
        _context(self)
        if not isinstance(self.candidates, (tuple, list)) or any(
                not isinstance(c, PhysicalConsequenceCandidate) for c in self.candidates):
            raise ValueError('candidates must contain PhysicalConsequenceCandidate records')
        ids, semantic = set(), set()
        for c in self.candidates:
            if any(getattr(c, n) != getattr(self, n) for n in
                   ('scene_id', 'timestamp', 'session_id', 'coordinate_frame_id')):
                raise ValueError('candidate must match bundle context')
            key = (c.modality, c.event_family, c.object_ids, c.source_physical_state_id,
                   c.source_constraint_ids, c.source_field_references, c.rule_id)
            if c.consequence_id in ids or key in semantic:
                raise ValueError('duplicate consequence ID or semantic candidate')
            ids.add(c.consequence_id)
            semantic.add(key)
        object.__setattr__(self, 'candidates', tuple(sorted(self.candidates,
            key=lambda c: (c.modality.value, c.event_family, c.object_ids, c.consequence_id))))
        object.__setattr__(self, 'provenance', _metadata(self.provenance))
        object.__setattr__(self, 'uncertainty', _strings(self.uncertainty, 'uncertainty'))
        bounded_plain(self)


# Only exact signals emitted by LatentPhysicalStateBuilder v0.1 are recognized.
# Static contact does not establish an impulse, a sound, friction or heat.
_RULES = {
    'contact_candidate': (('tactile_force', 'sustained_contact_force'),),
    'collision_risk_geometry': (('acoustic', 'impact'), ('tactile_force', 'contact_impulse'),
                                ('kinesthetic', 'abrupt_motion_change')),
    'tracking_or_motion_discontinuity': (('kinesthetic', 'acceleration_change'),
                                         ('tactile_force', 'inertial_force_change')),
}
_CONSTRAINT_KEYS = {
    'collision_risk_geometry': {('collision_possibility', 'collision_possible', 'satisfied')},
    'tracking_or_motion_discontinuity': {
        ('implausible_displacement', 'kinematic_outlier', 'violated'),
        ('inertia_consistency', 'abrupt_change_detected', 'violated')},
}
_LIMITS = ('consequence_not_observation', 'event_occurrence_unconfirmed',
           'sensor_availability_not_evaluated', 'metric_magnitude_unknown')


class CrossModalConsequenceBuilder:
    """Build from explicit latent signals; optional constraints audit their sources.

    No rule runs on bare constraint status, semantic classes or learned metadata.
    Missing/unsupported prerequisites are skipped, never treated as negative facts.
    """

    def build(self, physical_state, physics_constraints=None):
        if isinstance(physical_state, HybridWorldState):
            if physics_constraints is not None:
                raise ValueError('supply hybrid or standalone components, not both')
            hybrid = physical_state
        elif isinstance(physical_state, LatentPhysicalState):
            hybrid = HybridWorldState(physical_state, physics_constraints)
        else:
            raise ValueError('expected LatentPhysicalState or HybridWorldState')
        # Retain frozen HybridWorldState's schema and recursive context protection.
        HybridWorldState(hybrid.physical_state, hybrid.physics_constraints, hybrid.learned_signal)
        state, supplied = hybrid.physical_state, hybrid.physics_constraints
        assessment = supplied if isinstance(supplied, PhysicsTransitionAssessment) else None
        if assessment is not None and assessment.assessment_kind != 'retrospective_transition_ending_at_current_snapshot':
            raise ValueError('only retrospective transition assessments supported')
        bundle = assessment.constraints if assessment is not None else supplied
        bounded_plain(state)
        active = {}
        for item in state.active_constraints:
            identity = item['constraint_id']
            if identity in active and active[identity] != item:
                raise ValueError('conflicting active constraint identity')
            active[identity] = item
        results = {} if bundle is None else {r.constraint_id: r for r in bundle.results}
        objects = {o['physical_object_id']: o for o in state.objects}
        context = {n: getattr(state, n) for n in ('scene_id', 'timestamp', 'session_id', 'coordinate_frame_id')}
        uncertainty = set(state.uncertainty) | set(_LIMITS)
        if bundle is not None:
            for result in bundle.results:
                uncertainty.update(result.uncertainty)
        if state.timestamp is None:
            uncertainty.add('temporal_alignment_unknown')
        candidates = {}
        for section in ('relations', 'dynamics'):
            for source in getattr(state, section):
                signal = source.get('signal')
                if signal not in _RULES or (section == 'dynamics') != (signal == 'tracking_or_motion_discontinuity'):
                    continue
                refs = _strings(source.get('derived_from', ()), 'derived_from')
                status = source.get('status')
                if not refs or status not in ('possible', 'indeterminate'):
                    continue
                if section == 'relations' and not (source.get('value') is True or source.get('value') == 'possible'):
                    continue
                ids = _strings((source['subject_id'], source['object_id']) if section == 'relations'
                               else source.get('object_ids', ()), 'object_ids')
                if not ids or not set(ids) <= objects.keys() or (section == 'relations' and len(ids) != 2):
                    continue
                cids = tuple(r for r in refs if r in active or r in results)
                # Constraint-derived signals need a traceable matching assessment summary.
                if signal in _CONSTRAINT_KEYS and not cids:
                    continue
                local_uncertainty = uncertainty | set(source.get('uncertainty', ()))
                for oid in ids:
                    local_uncertainty.update(objects[oid].get('uncertainty', ()))
                selected = {}
                skip = False
                for cid in cids:
                    summaries = ([active[cid]] if cid in active else [])
                    if cid in results:
                        summaries.append(bounded_plain(results[cid]))
                        local_uncertainty.update(results[cid].uncertainty)
                    selected[cid] = summaries
                    for summary in summaries:
                        if tuple(sorted(summary['object_ids'])) != ids:
                            raise ValueError('source constraint object mismatch')
                        key = tuple(summary[n] for n in ('constraint_type', 'finding', 'status'))
                        if summary['status'] in ('unsupported', 'not_applicable'):
                            skip = True
                        elif signal in _CONSTRAINT_KEYS and key not in _CONSTRAINT_KEYS[signal]:
                            status = 'indeterminate'
                            local_uncertainty.add('source_constraint_conflict_or_ambiguity')
                if skip:
                    continue
                if signal == 'tracking_or_motion_discontinuity':
                    status = 'indeterminate'
                    local_uncertainty.add('tracking_or_camera_change_may_explain_signal')
                elif signal == 'contact_candidate':
                    local_uncertainty.add('contact_duration_and_load_unknown')
                else:
                    local_uncertainty.add('depth_and_actual_contact_unknown')
                rule_id = 'conditional_' + signal
                source_ref = stable_id('latent-source', section, signal, ids, refs)
                field_refs = tuple(sorted(set(refs + (state.latent_state_id + '.' + section + '[' + source_ref + ']',)
                    + tuple(ref for cid in cids if cid in results for ref in results[cid].derived_from))))
                provenance = {
                    'rule_id': rule_id, 'rule_version': VERSION,
                    'source_schema_version': state.schema_version,
                    'source_physical_state_id': state.latent_state_id,
                    'source_section': section, 'source_signal': signal,
                    'source_references': refs, 'source_status': source['status'],
                    'source_constraints': selected,
                    'source_state_provenance': state.provenance,
                    'assessment_id': None if assessment is None else assessment.assessment_id,
                    'assessment_kind': None if assessment is None else assessment.assessment_kind,
                    'assessment_provenance': None if assessment is None else assessment.provenance,
                    'constraint_schema_version': None if not cids else 'physics-constraints-0.2',
                    'truth_decision': 'not_performed', **context,
                }
                for modality, event in _RULES[signal]:
                    identity = stable_id('consequence', VERSION, context, state.latent_state_id,
                                         modality, event, ids, cids, refs, rule_id)
                    description = (f'{event} consequence {status}, conditional on actual physical interaction'
                                   if signal != 'tracking_or_motion_discontinuity' else
                                   f'{event} consequence indeterminate; physical motion change is unresolved')
                    candidate = PhysicalConsequenceCandidate(identity, **context, modality=modality,
                        event_family=event, status=status, object_ids=ids,
                        source_physical_state_id=state.latent_state_id, source_constraint_ids=cids,
                        source_field_references=field_refs, description=description, rule_id=rule_id,
                        provenance=provenance, uncertainty=tuple(sorted(local_uncertainty)))
                    if identity in candidates and candidates[identity] != candidate:
                        raise ValueError('conflicting duplicate source signal')
                    candidates[identity] = candidate
        return CrossModalConsequenceBundle(**context, candidates=tuple(candidates.values()),
            provenance={'builder': 'CrossModalConsequenceBuilder', 'rule_version': VERSION,
                        'source_physical_state_id': state.latent_state_id,
                        'source_schema_version': state.schema_version,
                        'source_provenance': state.provenance,
                        'learned_signal_policy': 'ignored_for_generation',
                        'missing_prerequisite_policy': 'skip_not_negative_evidence',
                        'truth_decision': 'not_performed'}, uncertainty=tuple(sorted(uncertainty)))
