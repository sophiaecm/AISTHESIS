"""Explicit conditional continuations; no sensing, branch ranking or belief update."""
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
import json

from ._structured import freeze, identifier, number
from .common_evidence_state import bounded_plain
from .cross_modal_consequences import CrossModalConsequenceBuilder, CrossModalConsequenceBundle
from .evidence import stable_id
from .hybrid_world_state import HybridWorldState, _alignment
from .latent_physical_state import LatentPhysicalState
from .physics_constraints import PhysicsTransitionAssessment


VERSION = 'multiple-futures-0.2'
_CONTEXT = ('scene_id', 'timestamp', 'session_id', 'coordinate_frame_id')
_LIMITS = ('future_is_hypothesis', 'branches_not_exhaustive', 'horizon_duration_unknown',
           'no_branch_is_not_impossibility', 'metric_future_state_unknown')


class FutureBranchStatus(str, Enum):
    POSSIBLE = 'possible'
    INDETERMINATE = 'indeterminate'
    UNSUPPORTED = 'unsupported'
    UNAVAILABLE = 'unavailable'


class FutureRelation(str, Enum):
    CO_POSSIBLE = 'co_possible'
    MUTUALLY_EXCLUSIVE = 'mutually_exclusive'
    UNRESOLVED_RELATION = 'unresolved_relation'


class FutureHorizonKind(str, Enum):
    NEXT_TRANSITION = 'next_transition'
    SHORT_HORIZON = 'short_horizon'


def _strings(values, name, required=False):
    if not isinstance(values, (tuple, list)):
        raise ValueError(f'{name} requires an ordered sequence')
    for value in values:
        identifier(value, name)
    if required and not values:
        raise ValueError(f'{name} must not be empty')
    return tuple(sorted(set(values)))


def _context(value):
    for name in ('scene_id', 'session_id', 'coordinate_frame_id'):
        identifier(getattr(value, name), name, optional=name == 'coordinate_frame_id')
    if value.timestamp is not None:
        number(value.timestamp, 'timestamp', nonnegative=True)


def _metadata(value):
    if not isinstance(value, Mapping) or not value:
        raise ValueError('nonempty structured provenance required')
    return freeze(bounded_plain(value))


def _horizon(kind, value):
    try:
        kind = FutureHorizonKind(kind)
    except (ValueError, TypeError) as exc:
        raise ValueError('invalid horizon kind') from exc
    if value is not None:
        raise ValueError('v0.2 supports qualitative horizons only; horizon_value must be None')
    return kind


class _Serializable:
    def to_dict(self):
        return bounded_plain(self)

    def to_json(self):
        return json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'), allow_nan=False)


@dataclass(frozen=True)
class FutureAssumption(_Serializable):
    assumption_id: str
    assumption_type: str
    source_refs: tuple[str, ...]
    statement: str

    def __post_init__(self):
        identifier(self.assumption_id, 'assumption_id')
        identifier(self.statement, 'statement')
        if self.assumption_type not in ('motion_persists', 'contact_within_horizon',
                'no_contact_within_horizon', 'continuity_resumes', 'outcome_unresolved'):
            raise ValueError('unsupported assumption type')
        object.__setattr__(self, 'source_refs', _strings(self.source_refs, 'source_refs', True))
        bounded_plain(self)


_CHANGES = {'motion_state': ('moving', 'unresolved'),
            'contact_status': ('possible_contact', 'no_contact_assumed', 'unresolved'),
            'continuity_status': ('possible_continuity', 'unresolved')}


@dataclass(frozen=True)
class FutureStateCandidate(_Serializable):
    future_id: str
    scene_id: str
    timestamp: float | None
    session_id: str
    coordinate_frame_id: str | None
    horizon_kind: FutureHorizonKind
    horizon_value: float | int | None
    status: FutureBranchStatus
    branch_family: str
    object_ids: tuple[str, ...]
    source_physical_state_id: str
    source_constraint_ids: tuple[str, ...]
    source_consequence_ids: tuple[str, ...]
    source_field_references: tuple[str, ...]
    assumptions: tuple[FutureAssumption, ...]
    predicted_changes: tuple[Mapping, ...]
    uncertainty: tuple[str, ...]
    provenance: Mapping
    rule_id: str
    confidence: float | None = None
    rule_version: str = field(default=VERSION, init=False)
    schema_version: str = field(default=VERSION, init=False)

    def __post_init__(self):
        _context(self)
        for name in ('future_id', 'source_physical_state_id', 'rule_id'):
            identifier(getattr(self, name), name)
        if self.branch_family not in ('motion_continuation', 'contact_resolution', 'discontinuity_resolution'):
            raise ValueError('unsupported branch family')
        object.__setattr__(self, 'horizon_kind', _horizon(self.horizon_kind, self.horizon_value))
        try:
            object.__setattr__(self, 'status', FutureBranchStatus(self.status))
        except (ValueError, TypeError) as exc:
            raise ValueError('invalid future status') from exc
        if self.confidence is not None:
            raise ValueError('v0.2 does not assign branch probabilities or confidence scores')
        for name in ('object_ids', 'source_constraint_ids', 'source_consequence_ids',
                     'source_field_references', 'uncertainty'):
            object.__setattr__(self, name, _strings(getattr(self, name), name,
                name in ('object_ids', 'source_field_references')))
        if not isinstance(self.assumptions, (tuple, list)) or not self.assumptions or any(
                not isinstance(a, FutureAssumption) for a in self.assumptions):
            raise ValueError('branches require structured assumptions')
        if len({a.assumption_id for a in self.assumptions}) != len(self.assumptions):
            raise ValueError('duplicate assumption ID')
        if any(not set(a.source_refs) <= set(self.source_field_references) for a in self.assumptions):
            raise ValueError('assumption references must belong to branch sources')
        types = {a.assumption_type for a in self.assumptions}
        if {'contact_within_horizon', 'no_contact_within_horizon'} <= types:
            raise ValueError('branch has contradictory contact assumptions')
        object.__setattr__(self, 'assumptions', tuple(sorted(self.assumptions, key=lambda a: a.assumption_id)))
        if not isinstance(self.predicted_changes, (tuple, list)) or not self.predicted_changes:
            raise ValueError('structured predicted changes required')
        changes = {}
        consequence_refs = set()
        for change in self.predicted_changes:
            if not isinstance(change, Mapping) or set(change) != {'field', 'value', 'epistemic_status'}:
                raise ValueError('change requires field, value and epistemic_status only')
            name, value, status = (change[n] for n in ('field', 'value', 'epistemic_status'))
            if any(not isinstance(v, str) for v in (name, value, status)):
                raise ValueError('conditional change fields must be strings')
            if name == 'consequence_reference':
                if value not in self.source_consequence_ids or status not in ('expected', 'possible', 'indeterminate'):
                    raise ValueError('consequence change requires a candidate source and non-observational status')
                consequence_refs.add(value)
            elif name not in _CHANGES or value not in _CHANGES[name] or status != 'branch_assumption':
                raise ValueError('unsupported conditional change')
            key = (name, value)
            if key in changes:
                raise ValueError('duplicate predicted change')
            if name != 'consequence_reference' and any(k[0] == name for k in changes):
                raise ValueError('conflicting changes for one physical field')
            changes[key] = freeze(bounded_plain(change))
        if consequence_refs != set(self.source_consequence_ids):
            raise ValueError('every consequence source requires an explicit candidate reference')
        object.__setattr__(self, 'predicted_changes', tuple(changes[k] for k in sorted(changes)))
        object.__setattr__(self, 'provenance', _metadata(self.provenance))
        bounded_plain(self)


@dataclass(frozen=True)
class FutureBranchRelation(_Serializable):
    relation_id: str
    branch_ids: tuple[str, str]
    relation: FutureRelation
    basis: str

    def __post_init__(self):
        identifier(self.relation_id, 'relation_id')
        identifier(self.basis, 'basis')
        ids = _strings(self.branch_ids, 'branch_ids', True)
        if len(ids) != 2:
            raise ValueError('relation requires two distinct branches')
        object.__setattr__(self, 'branch_ids', ids)
        try:
            object.__setattr__(self, 'relation', FutureRelation(self.relation))
        except (ValueError, TypeError) as exc:
            raise ValueError('invalid branch relation') from exc
        bounded_plain(self)


def _exclusive(a, b):
    """Only opposite contact assumptions for the same endpoints and horizon."""
    if a.branch_family != 'contact_resolution' or b.branch_family != 'contact_resolution':
        return False
    if a.object_ids != b.object_ids or (a.horizon_kind, a.horizon_value) != (b.horizon_kind, b.horizon_value):
        return False
    left, right = ({x.assumption_type for x in c.assumptions} for c in (a, b))
    return (('contact_within_horizon' in left and 'no_contact_within_horizon' in right)
            or ('no_contact_within_horizon' in left and 'contact_within_horizon' in right))


@dataclass(frozen=True)
class MultipleFutureBundle(_Serializable):
    scene_id: str
    timestamp: float | None
    session_id: str
    coordinate_frame_id: str | None
    source_physical_state_id: str
    branches: tuple[FutureStateCandidate, ...]
    branch_relations: tuple[FutureBranchRelation, ...]
    uncertainty: tuple[str, ...]
    provenance: Mapping
    schema_version: str = field(default=VERSION, init=False)

    def __post_init__(self):
        _context(self)
        identifier(self.source_physical_state_id, 'source_physical_state_id')
        if not isinstance(self.branches, (tuple, list)) or any(
                not isinstance(b, FutureStateCandidate) for b in self.branches):
            raise ValueError('branches must contain FutureStateCandidate records')
        branches, identities = {}, set()
        for branch in self.branches:
            if any(getattr(branch, n) != getattr(self, n) for n in _CONTEXT + ('source_physical_state_id',)):
                raise ValueError('branch must match bundle source and context')
            key = (branch.branch_family, branch.object_ids, branch.horizon_kind, branch.horizon_value,
                   branch.source_constraint_ids, branch.source_consequence_ids, branch.source_field_references,
                   tuple((a.assumption_type, a.source_refs) for a in branch.assumptions), branch.rule_id)
            if branch.future_id in branches or key in identities:
                raise ValueError('duplicate branch ID or semantic branch')
            branches[branch.future_id] = branch
            identities.add(key)
        if not isinstance(self.branch_relations, (tuple, list)) or any(
                not isinstance(r, FutureBranchRelation) for r in self.branch_relations):
            raise ValueError('branch_relations requires FutureBranchRelation records')
        seen_ids, seen_pairs = set(), set()
        for relation in self.branch_relations:
            if not set(relation.branch_ids) <= branches.keys():
                raise ValueError('branch relation references missing branch')
            if relation.relation_id in seen_ids or relation.branch_ids in seen_pairs:
                raise ValueError('duplicate branch relation ID or pair')
            a, b = (branches[i] for i in relation.branch_ids)
            exclusive = _exclusive(a, b)
            if relation.relation == 'mutually_exclusive' and not exclusive:
                raise ValueError('exclusivity requires opposite contact assumptions for the same scope')
            if relation.relation == 'co_possible' and exclusive:
                raise ValueError('opposite contact assumptions cannot be co_possible')
            seen_ids.add(relation.relation_id)
            seen_pairs.add(relation.branch_ids)
        object.__setattr__(self, 'branches', tuple(sorted(branches.values(), key=lambda b:
            (b.branch_family, b.object_ids, tuple(a.assumption_type for a in b.assumptions), b.future_id))))
        object.__setattr__(self, 'branch_relations', tuple(sorted(self.branch_relations, key=lambda r: r.branch_ids)))
        object.__setattr__(self, 'uncertainty', _strings(self.uncertainty, 'uncertainty'))
        object.__setattr__(self, 'provenance', _metadata(self.provenance))
        bounded_plain(self)


_OPTIONS = {
    'motion_continuation': (
        ('motion_persists', 'current motion persists over the represented horizon', 'motion_state', 'moving'),
        ('outcome_unresolved', 'motion continuation remains unresolved', 'motion_state', 'unresolved')),
    'contact_resolution': (
        ('contact_within_horizon', 'candidate geometry resolves to contact within the represented horizon',
         'contact_status', 'possible_contact'),
        ('no_contact_within_horizon', 'candidate geometry does not resolve to contact within the represented horizon',
         'contact_status', 'no_contact_assumed'),
        ('outcome_unresolved', 'contact outcome remains unresolved over the represented horizon',
         'contact_status', 'unresolved')),
    'discontinuity_resolution': (
        ('continuity_resumes', 'sampled continuity resumes without assigning a cause to the discontinuity',
         'continuity_status', 'possible_continuity'),
        ('outcome_unresolved', 'motion discontinuity remains unresolved', 'continuity_status', 'unresolved')),
}


class MultipleFuturesBuilder:
    """Local continuation hypotheses, not mutually exclusive whole-world scenarios.

    Step 17's public builder audits explicit contact/discontinuity prerequisites.
    Its internal candidates are not attached unless supplied by the caller.
    """

    def build(self, physical_state, physics_constraints=None, consequences=None, *,
              horizon_kind='next_transition', horizon_value=None):
        horizon_kind = _horizon(horizon_kind, horizon_value)
        if isinstance(physical_state, HybridWorldState):
            if physics_constraints is not None:
                raise ValueError('supply hybrid or standalone constraints, not both')
            hybrid = HybridWorldState(physical_state.physical_state, physical_state.physics_constraints,
                                     physical_state.learned_signal)
        elif isinstance(physical_state, LatentPhysicalState):
            hybrid = HybridWorldState(physical_state, physics_constraints)
        else:
            raise ValueError('expected LatentPhysicalState or HybridWorldState')
        state, constraints = hybrid.physical_state, hybrid.physics_constraints
        # Public Step 17 API preserves exact vocabulary, unsupported-source skips,
        # conflict downgrades, endpoint checks and retrospective assessment semantics.
        audit = CrossModalConsequenceBuilder().build(hybrid)
        context = {n: getattr(state, n) for n in _CONTEXT}
        objects = {o['physical_object_id']: o for o in state.objects}
        source_bundle = constraints.constraints if isinstance(constraints, PhysicsTransitionAssessment) else constraints
        uncertainty = set(state.uncertainty) | set(_LIMITS)
        if state.timestamp is None:
            uncertainty.add('temporal_alignment_unknown')
        if source_bundle is None:
            uncertainty.add('physics_constraints_unavailable')
        else:
            for result in source_bundle.results:
                uncertainty.update(result.uncertainty)
        supplied = {}
        if consequences is None:
            uncertainty.add('cross_modal_consequences_unavailable')
        else:
            if not isinstance(consequences, CrossModalConsequenceBundle) or consequences.schema_version != 'cross-modal-consequences-0.2':
                raise ValueError('expected CrossModalConsequenceBundle v0.2')
            if any(getattr(consequences, n) != context[n] for n in _CONTEXT):
                raise ValueError('consequence bundle context mismatch')
            data = bounded_plain(consequences)
            _alignment(data, state)
            declared_source = consequences.provenance.get('source_physical_state_id')
            if declared_source is not None and declared_source != state.latent_state_id:
                raise ValueError('consequence bundle physical source mismatch')
            uncertainty.update(consequences.uncertainty)
            for candidate in consequences.candidates:
                if candidate.source_physical_state_id != state.latent_state_id:
                    raise ValueError('consequence physical source mismatch')
                if not set(candidate.object_ids) <= objects.keys():
                    raise ValueError('consequence refers to absent physical object')
                supplied[candidate.consequence_id] = candidate

        # Collapse modalities from the SAME physical source, not independent votes.
        groups = {}
        for candidate in audit.candidates:
            p = candidate.provenance
            signal = p['source_signal']
            family = 'discontinuity_resolution' if signal == 'tracking_or_motion_discontinuity' else 'contact_resolution'
            key = (family, signal, candidate.object_ids, candidate.source_constraint_ids, p['source_references'])
            if key not in groups:
                groups[key] = {'family': family, 'signal': signal, 'ids': candidate.object_ids,
                    'constraint_ids': candidate.source_constraint_ids, 'refs': candidate.source_field_references,
                    'status': candidate.status.value, 'uncertainty': set(candidate.uncertainty),
                    'source': p, 'consequences': []}
            group = groups[key]
            group['uncertainty'].update(candidate.uncertainty)
            attached = supplied.get(candidate.consequence_id)
            if attached is not None:
                # Identity alone is insufficient: verify the source and rule binding.
                names = ('modality', 'event_family', 'object_ids', 'source_constraint_ids', 'rule_id', 'rule_version')
                if any(getattr(attached, n) != getattr(candidate, n) for n in names):
                    raise ValueError('consequence identity conflicts with explicit source binding')
                # Full optional constraints can add field references without changing
                # Step 17's semantic ID. Compare the original source selector instead.
                if (attached.provenance.get('source_signal') != signal or
                        attached.provenance.get('source_references') != p['source_references'] or
                        not set(p['source_references']) <= set(attached.source_field_references)):
                    raise ValueError('consequence source references conflict with explicit source')
                if attached.status in ('expected', 'possible', 'indeterminate'):
                    group['consequences'].append(attached)
                    group['uncertainty'].update(attached.uncertainty)

        for obj in state.objects:
            motion = obj.get('attributes', {}).get('motion_state', {})
            if motion.get('value') != 'moving' or motion.get('status') not in ('observed', 'estimated'):
                continue
            refs = _strings(motion.get('derived_from', ()), 'motion source references')
            if not refs:
                continue
            oid = obj['physical_object_id']
            refs = tuple(sorted(set(refs + (f'{state.latent_state_id}.objects[{oid!r}].attributes.motion_state',))))
            groups[('motion_continuation', oid)] = {'family': 'motion_continuation', 'signal': 'motion_state',
                'ids': (oid,), 'constraint_ids': (), 'refs': refs, 'status': 'possible',
                'uncertainty': set(obj.get('uncertainty', ())), 'source': {'motion_state': motion}, 'consequences': []}

        branches, used_consequences = {}, set()
        for group in groups.values():
            family, ids, refs = group['family'], group['ids'], group['refs']
            for kind, statement, change_field, change_value in _OPTIONS[family]:
                # A no-contact assumption does not inherit an impact consequence.
                attached = group['consequences'] if kind in ('contact_within_horizon', 'outcome_unresolved') else []
                cids = tuple(sorted(c.consequence_id for c in attached))
                used_consequences.update(cids)
                assumption = FutureAssumption(stable_id('future-assumption', VERSION, context, horizon_kind,
                    state.latent_state_id, family, ids, refs, kind), kind, refs, statement)
                rule_id = 'branch_' + group['signal']
                status = 'indeterminate' if kind == 'outcome_unresolved' or group['status'] == 'indeterminate' else 'possible'
                changes = [{'field': change_field, 'value': change_value, 'epistemic_status': 'branch_assumption'}]
                changes.extend({'field': 'consequence_reference', 'value': c.consequence_id,
                                'epistemic_status': c.status.value} for c in attached)
                values = dict(**context, horizon_kind=horizon_kind, horizon_value=None, status=status,
                    branch_family=family, object_ids=ids, source_physical_state_id=state.latent_state_id,
                    source_constraint_ids=group['constraint_ids'], source_consequence_ids=cids,
                    source_field_references=refs, assumptions=(assumption,), predicted_changes=tuple(changes),
                    uncertainty=tuple(sorted(uncertainty | group['uncertainty'])), rule_id=rule_id,
                    provenance={'rule_id': rule_id, 'rule_version': VERSION, 'source_signal': group['signal'],
                        'source_schema_version': state.schema_version, 'source_details': group['source'],
                        'source_state_provenance': state.provenance,
                        'consequence_sources': {c.consequence_id: c.to_dict() for c in attached},
                        'consequence_schema_version': None if not attached else consequences.schema_version,
                        'assumption_ids': (assumption.assumption_id,), 'truth_decision': 'not_performed'})
                identity = stable_id('future', VERSION, context, state.latent_state_id, horizon_kind,
                    family, ids, group['constraint_ids'], cids, refs, assumption.to_dict(), rule_id)
                branch = FutureStateCandidate(identity, **values)
                if identity in branches and branches[identity] != branch:
                    raise ValueError('conflicting source branches')
                branches[identity] = branch

        relations = []
        for a, b in combinations(sorted(branches.values(), key=lambda b: b.future_id), 2):
            exclusive = _exclusive(a, b)
            relation = 'mutually_exclusive' if exclusive else 'unresolved_relation'
            basis = 'opposite_contact_assumptions_same_objects_and_horizon' if exclusive else 'joint_compatibility_not_evaluated'
            pair = (a.future_id, b.future_id)
            relations.append(FutureBranchRelation(stable_id('future-relation', VERSION, pair, relation), pair, relation, basis))
        if supplied.keys() - used_consequences:
            uncertainty.add('unassociated_consequence_candidates')
        if not branches:
            uncertainty.add('no_eligible_explicit_branch_source')
        return MultipleFutureBundle(**context, source_physical_state_id=state.latent_state_id,
            branches=tuple(branches.values()), branch_relations=tuple(relations), uncertainty=tuple(sorted(uncertainty)),
            provenance={'builder': 'MultipleFuturesBuilder', 'rule_version': VERSION,
                'source_schema_version': state.schema_version, 'source_physical_state_id': state.latent_state_id,
                'source_state_provenance': state.provenance,
                'constraint_source': None if constraints is None else bounded_plain(constraints),
                'consequence_bundle_provenance': None if consequences is None else consequences.provenance,
                'unassociated_consequence_ids': tuple(sorted(supplied.keys() - used_consequences)),
                'horizon_kind': horizon_kind.value, 'horizon_value': None,
                'missing_prerequisite_policy': 'skip_not_impossible', 'learned_signal_policy': 'ignored_for_generation',
                'branch_scope': 'local_hypotheses_not_joint_world_scenarios',
                'ordering': 'semantic_keys_only', 'truth_decision': 'not_performed'})
