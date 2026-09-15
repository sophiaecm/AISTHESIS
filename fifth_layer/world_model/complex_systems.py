"""Conservative systems structure over explicit physical sources; no causal discovery."""
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
import json

from ._structured import freeze, identifier, number
from .common_evidence_state import bounded_plain
from .evidence import stable_id
from .hybrid_world_state import _alignment
from .latent_physical_state import LatentPhysicalState
from .physics_constraints import PhysicsConstraintBundle, PhysicsTransitionAssessment
from .cross_modal_consequences import CrossModalConsequenceBundle
from .multiple_futures import MultipleFutureBundle

VERSION = 'complex-systems-v0.1'
TYPES = {'contact_candidate': 'contact_relation_candidate',
         'collision_risk_geometry': 'collision_relation_candidate',
         'support_candidate': 'support_relation_candidate',
         'tracking_or_motion_discontinuity': 'discontinuity_relation_candidate'}
STATUSES = ('present_candidate', 'possible', 'indeterminate', 'unsupported', 'unavailable')
VALID = ('present_candidate', 'possible', 'indeterminate')
LIMITS = ('relation_direction_unknown', 'subsystem_causality_unknown',
          'missing_interaction_not_independence')
POLICY = dict(causal_discovery=False, hidden_actor_inference=False, branch_selection=False,
    Bayesian_feedback=False, experience_learning=False, current_observation=False,
    global_complexity_score=False, cascade='deferred', propagation='deferred',
    interpretation='engineering_candidate_structure_not_scientific_validation')


def _strings(values):
    if not isinstance(values, (tuple, list)):
        raise ValueError('ordered string sequence required')
    for value in values:
        identifier(value, 'reference')
    return tuple(sorted(set(values)))


class _Record:
    def to_dict(self):
        return bounded_plain(self)

    def to_json(self):
        return json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'), allow_nan=False)

    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name in ('scene_id', 'session_id', 'coordinate_frame_id', 'source_physical_state_id', 'subsystem_id') and f.init:
                identifier(value, f.name)
            if f.name == 'timestamp' and value is not None:
                number(value, 'timestamp', nonnegative=True)
            if isinstance(value, Mapping):
                object.__setattr__(self, f.name, freeze(bounded_plain(value)))
            if f.name in ('object_ids', 'interaction_ids', 'source_constraint_ids', 'source_consequence_ids',
                          'source_future_ids', 'source_field_references', 'source_refs', 'uncertainty'):
                object.__setattr__(self, f.name, _strings(value))
        id_name = next(f.name for f in fields(self) if not f.init and f.name.endswith('_id'))
        data = self.to_dict()
        data.pop(id_name)
        object.__setattr__(self, id_name, stable_id(id_name, VERSION, data))


@dataclass(frozen=True)
class InteractionEdge(_Record):
    scene_id: str
    timestamp: float | None
    session_id: str
    coordinate_frame_id: str
    object_ids: tuple
    interaction_type: str
    status: str
    source_constraint_ids: tuple
    source_consequence_ids: tuple
    source_future_ids: tuple
    source_field_references: tuple
    uncertainty: tuple
    provenance: Mapping
    interaction_id: str = field(default='', init=False)
    schema_version: str = field(default=VERSION, init=False)

    def __post_init__(self):
        if self.interaction_type not in (*TYPES.values(), 'shared_constraint_relation') or self.status not in STATUSES:
            raise ValueError('unsupported interaction vocabulary')
        if len(set(self.object_ids)) < 2 or not self.source_field_references:
            raise ValueError('interaction requires multiple objects and explicit sources')
        super().__post_init__()


@dataclass(frozen=True)
class CoupledSubsystemCandidate(_Record):
    scene_id: str
    timestamp: float | None
    session_id: str
    coordinate_frame_id: str
    object_ids: tuple
    interaction_ids: tuple
    coupling_basis: str
    status: str
    uncertainty: tuple
    provenance: Mapping
    subsystem_id: str = field(default='', init=False)
    schema_version: str = field(default=VERSION, init=False)

    def __post_init__(self):
        if len(set(self.object_ids)) < 2 or not self.interaction_ids or self.status not in ('possible', 'indeterminate'):
            raise ValueError('subsystem requires connected candidate sources and multiple objects')
        super().__post_init__()


@dataclass(frozen=True)
class StabilityAssessmentCandidate(_Record):
    subsystem_id: str
    scene_id: str
    timestamp: float | None
    status: str
    indicators: Mapping
    counter_indicators: Mapping
    source_refs: tuple
    uncertainty: tuple
    provenance: Mapping
    assessment_id: str = field(default='', init=False)
    schema_version: str = field(default=VERSION, init=False)

    def __post_init__(self):
        if self.status not in ('stable_candidate', 'unstable_candidate', 'mixed', 'indeterminate'):
            raise ValueError('unsupported stability status')
        super().__post_init__()


@dataclass(frozen=True)
class TransitionPatternCandidate(_Record):
    subsystem_id: str
    pattern_type: str
    status: str
    source_refs: tuple
    assumptions: Mapping
    uncertainty: tuple
    provenance: Mapping
    transition_id: str = field(default='', init=False)
    schema_version: str = field(default=VERSION, init=False)

    def __post_init__(self):
        if self.pattern_type not in ('contact_transition_candidate', 'collision_transition_candidate',
                                     'continuity_transition_candidate', 'unresolved_transition') or self.status not in STATUSES:
            raise ValueError('unsupported transition vocabulary')
        super().__post_init__()


@dataclass(frozen=True)
class ComplexSystemsState(_Record):
    scene_id: str
    timestamp: float | None
    session_id: str
    coordinate_frame_id: str
    source_physical_state_id: str
    interactions: tuple
    subsystems: tuple
    stability_assessments: tuple
    transition_candidates: tuple
    uncertainty: tuple
    provenance: Mapping
    complex_state_id: str = field(default='', init=False)
    schema_version: str = field(default=VERSION, init=False)

    def __post_init__(self):
        for name, cls, identity in (('interactions', InteractionEdge, 'interaction_id'),
                ('subsystems', CoupledSubsystemCandidate, 'subsystem_id'),
                ('stability_assessments', StabilityAssessmentCandidate, 'assessment_id'),
                ('transition_candidates', TransitionPatternCandidate, 'transition_id')):
            values = getattr(self, name)
            if not isinstance(values, (tuple, list)) or any(not isinstance(v, cls) for v in values):
                raise ValueError('typed ordered records required')
            if len({getattr(v, identity) for v in values}) != len(values):
                raise ValueError('duplicate output identity')
            object.__setattr__(self, name, tuple(sorted(values, key=lambda v: getattr(v, identity))))
        edges = {e.interaction_id: e for e in self.interactions}
        subsystem_ids = {s.subsystem_id for s in self.subsystems}
        for s in self.subsystems:
            if not set(s.interaction_ids) <= edges.keys() or any(not set(edges[i].object_ids) <= set(s.object_ids) for i in s.interaction_ids):
                raise ValueError('subsystem references invalid interactions')
        if any(a.subsystem_id not in subsystem_ids for a in (*self.stability_assessments, *self.transition_candidates)):
            raise ValueError('unknown subsystem reference')
        super().__post_init__()


def _status(status):
    return {'observed': 'present_candidate', 'estimated': 'present_candidate', 'possible': 'possible',
            'satisfied': 'present_candidate', 'violated': 'indeterminate', 'indeterminate': 'indeterminate',
            'unknown': 'indeterminate', 'unsupported': 'unsupported', 'unavailable': 'unavailable',
            'not_applicable': 'unavailable'}.get(status)


def _aligned(value, physical):
    for name in ('scene_id', 'session_id', 'timestamp', 'coordinate_frame_id'):
        if hasattr(value, name) and getattr(value, name) != getattr(physical, name):
            raise ValueError(f'{name} mismatch')
    _alignment(bounded_plain(value), physical)


class ComplexSystemsBuilder:
    def build(self, physical_state, physics_constraints=None, cross_modal_consequences=None, multiple_futures=None):
        p = physical_state
        if not isinstance(p, LatentPhysicalState) or p.schema_version != 'latent-physical-state-0.1':
            raise ValueError('LatentPhysicalState v0.1 required')
        replace(p)
        _aligned(p, p)
        context = {name: getattr(p, name) for name in ('scene_id', 'timestamp', 'session_id', 'coordinate_frame_id')}
        objects = {o['physical_object_id'] for o in p.objects}
        uncertainty = set(p.uncertainty) | set(LIMITS)
        for record in (*p.objects, *p.relations, *p.dynamics):
            uncertainty.update(record.get('uncertainty', ()))
        if p.timestamp is None:
            uncertainty.add('temporal_alignment_unknown')
        availability, source_data = {}, {'physical_state': bounded_plain(p)}
        for name, value, cls, version in (
            ('physics_constraints', physics_constraints, (PhysicsConstraintBundle, PhysicsTransitionAssessment), None),
            ('cross_modal_consequences', cross_modal_consequences, CrossModalConsequenceBundle, 'cross-modal-consequences-0.2'),
            ('multiple_futures', multiple_futures, MultipleFutureBundle, 'multiple-futures-0.2')):
            availability[name] = 'unavailable' if value is None else 'available'
            if value is None:
                uncertainty.add(name + '_unavailable')
                continue
            if not isinstance(value, cls):
                raise ValueError('invalid optional layer type')
            expected = version or ('physics-transition-0.2' if isinstance(value, PhysicsTransitionAssessment) else 'physics-constraints-0.2')
            if value.schema_version != expected:
                raise ValueError('unsupported source version')
            replace(value)
            _aligned(value, p)
            source_data[name] = bounded_plain(value)
            uncertainty.update(getattr(value, 'uncertainty', ()))
        bundle = physics_constraints.constraints if isinstance(physics_constraints, PhysicsTransitionAssessment) else physics_constraints
        constraints = {}
        for c in p.active_constraints:
            required = {'constraint_id', 'constraint_type', 'status', 'finding', 'object_ids'}
            if not isinstance(c, Mapping) or not required <= c.keys():
                raise ValueError('incomplete active constraint')
            c = dict(c)
            c['object_ids'] = _strings(c['object_ids'])
            identity = c['constraint_id']
            if identity in constraints and constraints[identity] != c:
                raise ValueError('conflicting active constraint ID')
            constraints[identity] = c
        if bundle:
            for r in bundle.results:
                replace(r)
                c = bounded_plain(r)
                c['object_ids'] = _strings(c['object_ids'])
                previous = constraints.get(r.constraint_id)
                if previous and any(previous[k] != c[k] for k in ('constraint_type', 'status', 'finding', 'object_ids')):
                    raise ValueError('constraint summary conflicts with supplied result')
                constraints[r.constraint_id] = c
                uncertainty.update(r.uncertainty)
        for c in constraints.values():
            if not set(c['object_ids']) <= objects:
                raise ValueError('constraint object absent from current physical state')
            if c['status'] == 'indeterminate':
                uncertainty.add('constraint_indeterminate')
        candidates = cross_modal_consequences.candidates if cross_modal_consequences else ()
        branches = multiple_futures.branches if multiple_futures else ()
        for value in (cross_modal_consequences, multiple_futures):
            if value is not None:
                ref = getattr(value, 'source_physical_state_id', value.provenance.get('source_physical_state_id'))
                if ref is not None and ref != p.latent_state_id:
                    raise ValueError('source physical state mismatch')
        for record in (*candidates, *branches):
            replace(record)
            if record.source_physical_state_id != p.latent_state_id or not set(record.object_ids) <= objects:
                raise ValueError('optional source physical/object mismatch')
            if not set(record.source_constraint_ids) <= constraints.keys():
                raise ValueError('unknown source constraint reference')
            uncertainty.update(record.uncertainty)
        if cross_modal_consequences:
            uncertainty.add('cross_modal_candidate_only')
            if any(not set(b.source_consequence_ids) <= {c.consequence_id for c in candidates} for b in branches):
                raise ValueError('unknown supplied consequence reference')
        if multiple_futures:
            uncertainty.add('future_branch_non_exhaustive')
        grouped, audit = {}, []
        def admit(kind, ids, status, source, refs, constraint_ids=()):
            ids = _strings(ids)
            if not set(ids) <= objects:
                raise ValueError('interaction references absent object')
            mapped = _status(status)
            if mapped is None or len(ids) < 2 or not refs:
                audit.append(dict(source=source, reason='unsupported_status_singleton_or_missing_references'))
                return
            key = (kind, ids, mapped)
            group = grouped.setdefault(key, dict(sources={}, refs=set(), constraints=set()))
            group['sources'][stable_id('interaction-source', source)] = source
            group['refs'].update(refs)
            group['constraints'].update(constraint_ids)
        for r in p.relations:
            signal = r.get('signal')
            if signal in TYPES:
                if r.get('value') is False or r.get('value') is None:
                    audit.append(dict(source=bounded_plain(r), reason='relation_value_negative_or_unknown'))
                    continue
                refs = _strings(r.get('derived_from', ()))
                admit(TYPES[signal], (r['subject_id'], r['object_id']), r.get('status'),
                    bounded_plain(r), refs, set(refs) & constraints.keys())
        for d in p.dynamics:
            if d.get('signal') == 'tracking_or_motion_discontinuity':
                refs = _strings(d.get('derived_from', ()))
                admit(TYPES[d['signal']], d['object_ids'], d.get('status'), bounded_plain(d), refs,
                      set(refs) & constraints.keys())
        for identity, c in sorted(constraints.items()):
            admit('shared_constraint_relation', c['object_ids'], c['status'], c,
                  (identity,) + tuple(c.get('derived_from', ())), (identity,))
        interactions = []
        for (kind, ids, status), group in sorted(grouped.items()):
            cs = tuple(c for c in candidates if set(c.object_ids) == set(ids))
            fs = tuple(b for b in branches if set(b.object_ids) == set(ids))
            edge_uncertainty = set(uncertainty)
            if status == 'indeterminate':
                edge_uncertainty.add('interaction_indeterminate')
            interactions.append(InteractionEdge(**context, object_ids=ids, interaction_type=kind, status=status,
                source_constraint_ids=tuple(sorted(group['constraints'])),
                source_consequence_ids=tuple(c.consequence_id for c in cs), source_future_ids=tuple(b.future_id for b in fs),
                source_field_references=tuple(sorted(group['refs'])), uncertainty=tuple(sorted(edge_uncertainty)),
                provenance=dict(POLICY, source_records=tuple(group['sources'][i] for i in sorted(group['sources'])),
                    consequence_context=bounded_plain(cs), future_context=bounded_plain(fs),
                    enrichment_policy='exact_object_scope_context_only', undirected=True)))
        subsystems = _components(interactions, context, uncertainty)
        assessments, transitions = [], []
        for subsystem in subsystems:
            edges = [e for e in interactions if e.interaction_id in subsystem.interaction_ids]
            assessments.append(_stability(subsystem, edges, constraints, p))
            for branch in branches:
                if not set(branch.object_ids) <= set(subsystem.object_ids):
                    continue
                kinds = {e.interaction_type for e in edges if set(e.object_ids) == set(branch.object_ids)}
                assumption_types = {a.assumption_type for a in branch.assumptions}
                patterns = []
                if branch.branch_family == 'contact_resolution' and kinds & {'contact_relation_candidate', 'collision_relation_candidate'}:
                    patterns.append('unresolved_transition' if 'outcome_unresolved' in assumption_types else 'contact_transition_candidate')
                    if 'contact_within_horizon' in assumption_types and 'collision_relation_candidate' in kinds:
                        patterns.append('collision_transition_candidate')
                if branch.branch_family == 'discontinuity_resolution' and any(
                    d.get('signal') == 'tracking_or_motion_discontinuity' and d.get('status') in ('possible', 'estimated', 'indeterminate')
                    and set(d['object_ids']) == set(branch.object_ids) for d in p.dynamics):
                    patterns.append('unresolved_transition' if 'outcome_unresolved' in assumption_types else 'continuity_transition_candidate')
                for pattern in patterns:
                    status = branch.status.value
                    if pattern == 'unresolved_transition' and status not in ('unsupported', 'unavailable'):
                        status = 'indeterminate'
                    transitions.append(TransitionPatternCandidate(subsystem.subsystem_id, pattern, status,
                        (branch.future_id,) + branch.source_field_references,
                        {'branch_assumptions': bounded_plain(branch.assumptions)}, subsystem.uncertainty,
                        dict(POLICY, source_future=bounded_plain(branch), occurred=False)))
        # Source snapshots retained as provenance, not evidence. Canonicalize the
        # unordered physical inventories so input tuple permutations replay identically.
        for name in ('objects', 'relations', 'dynamics', 'active_constraints'):
            records = source_data['physical_state'][name]
            source_data['physical_state'][name] = [v for _, v in sorted({stable_id('record', v): v for v in records}.items())]
        return ComplexSystemsState(**context, source_physical_state_id=p.latent_state_id,
            interactions=tuple(interactions), subsystems=tuple(subsystems), stability_assessments=tuple(assessments),
            transition_candidates=tuple(transitions), uncertainty=tuple(sorted(uncertainty)),
            provenance=dict(POLICY, layer_availability=availability, source_snapshots=source_data,
                extraction_audit=tuple(v for _, v in sorted({stable_id('audit', a): a for a in audit}.items())),
                counts=dict(interactions=len(interactions), subsystems=len(subsystems)), counts_are_audit_only=True))


def _components(interactions, context, uncertainty):
    neighbors = {}
    for e in interactions:
        if e.status not in VALID:
            continue
        for oid in e.object_ids:
            neighbors.setdefault(oid, set()).update(set(e.object_ids) - {oid})
    remaining, result = set(neighbors), []
    while remaining:
        pending, group = [min(remaining)], set()
        while pending:
            oid = pending.pop()
            if oid not in group:
                group.add(oid)
                pending.extend(sorted(neighbors[oid] - group, reverse=True))
        remaining -= group
        edges = [e for e in interactions if e.status in VALID and set(e.object_ids) <= group]
        uncertain = any(e.status == 'indeterminate' for e in edges)
        flags = set(uncertainty).union(*(e.uncertainty for e in edges))
        result.append(CoupledSubsystemCandidate(**context, object_ids=tuple(sorted(group)),
            interaction_ids=tuple(e.interaction_id for e in edges), coupling_basis='undirected_connected_candidate_interactions',
            status='indeterminate' if uncertain else 'possible', uncertainty=tuple(sorted(flags)), provenance=POLICY))
    return result


def _stability(subsystem, edges, constraints, physical):
    positive, negative, covered = {}, {}, set()
    scope = set(subsystem.object_ids)
    incomplete = any(e.status == 'indeterminate' for e in edges)
    for identity, c in sorted(constraints.items()):
        ids = set(c['object_ids'])
        if not ids or not ids <= scope:
            continue
        key = (c['constraint_type'], c['finding'], c['status'])
        if key in (('support_stability_possibility', 'support_geometry_possible', 'satisfied'),
                   ('inertia_consistency', 'motion_continuity_consistent', 'satisfied')):
            positive[identity] = c
            covered.update(ids)
        if key in (('inertia_consistency', 'abrupt_change_detected', 'violated'),
                   ('implausible_displacement', 'kinematic_outlier', 'violated')):
            negative[identity] = c
        incomplete |= c['status'] == 'indeterminate'
    for e in edges:
        if e.interaction_type in ('collision_relation_candidate', 'discontinuity_relation_candidate') and e.status in ('possible', 'present_candidate'):
            negative[e.interaction_id] = {'indicator': e.interaction_type, 'object_ids': e.object_ids}
    for d in physical.dynamics:
        if (d.get('signal') == 'tracking_or_motion_discontinuity' and d.get('status') in ('possible', 'estimated')
                and set(d['object_ids']) and set(d['object_ids']) <= scope):
            negative[stable_id('dynamics-source', bounded_plain(d))] = d
    status = ('mixed' if positive and negative else 'unstable_candidate' if negative else
              'stable_candidate' if positive and covered == scope and not incomplete else 'indeterminate')
    return StabilityAssessmentCandidate(subsystem.subsystem_id, subsystem.scene_id, subsystem.timestamp,
        status, positive, negative, tuple(sorted(set(positive) | set(negative))), subsystem.uncertainty,
        dict(POLICY, assessment_scope='explicit_geometry_or_sampled_continuity_indicators_not_mechanical_proof',
             full_positive_object_coverage=covered == scope, incomplete_indicator_context=incomplete))
