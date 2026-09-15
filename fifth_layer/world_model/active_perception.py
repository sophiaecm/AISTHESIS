"""Advisory symbolic reobservation candidates over explicit current uncertainty."""
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
import json

from ._structured import freeze, identifier, number
from .common_evidence_state import bounded_plain
from .evidence import stable_id
from .latent_physical_state import LatentPhysicalState
from .complex_systems import ComplexSystemsState
from .topological_world import TopologicalWorldState, _validate as validate_complex

VERSION = 'active-perception-v0.1'
CUES = ('indeterminate_relation', 'relation_change_ambiguity', 'component_reconfiguration',
        'subsystem_uncertainty', 'stability_uncertainty', 'transition_uncertainty', 'physical_state_uncertainty')
SCOPES = ('object', 'object_set', 'relation', 'subsystem')
PRIORITIES = ('high', 'medium', 'low', 'indeterminate')
LIMITS = ('target_not_truth', 'priority_not_probability_confidence_or_risk',
          'uncertainty_not_hidden_actor', 'missing_evidence_not_negative_evidence',
          'empty_plan_not_certainty', 'observation_request_not_control')
POLICY = dict(advisory_only=True, executable_control=False, hidden_actor_inference=False,
              causal_discovery=False, Bayesian_feedback=False, experience_learning_feedback=False,
              branch_selection=False, information_gain_measured=False, external_grounding=False,
              validation='engineering_contract_only')


def strings(values):
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
            if f.name.endswith('_ids') or f.name in ('uncertainty', 'relation_types'):
                object.__setattr__(self, f.name, strings(value))
            elif f.name == 'provenance':
                if not isinstance(value, Mapping):
                    raise ValueError('mapping provenance required')
                object.__setattr__(self, f.name, freeze(bounded_plain(value)))
            elif f.init and f.name.endswith('_id'):
                identifier(value, f.name, optional=f.name == 'source_physical_state_id')
        identity = next(f.name for f in fields(self) if not f.init and f.name.endswith('_id'))
        data = self.to_dict()
        data.pop(identity)
        object.__setattr__(self, identity, stable_id(identity, VERSION, data))


@dataclass(frozen=True)
class ObservationCue(_Record):
    cue_type: str
    object_ids: tuple
    source_ids: tuple
    epistemic_status: str
    uncertainty: tuple
    provenance: Mapping
    cue_id: str = field(default='', init=False)
    schema_version: str = field(default=VERSION, init=False)

    def __post_init__(self):
        if self.cue_type not in CUES or not self.object_ids or not self.source_ids:
            raise ValueError('explicit supported cue and references required')
        identifier(self.epistemic_status, 'epistemic_status')
        super().__post_init__()


@dataclass(frozen=True)
class ObservationTargetCandidate(_Record):
    target_scope: str
    object_ids: tuple
    relation_types: tuple
    cue_ids: tuple
    source_ids: tuple
    rationale_code: str
    priority_band: str
    uncertainty: tuple
    provenance: Mapping
    target_id: str = field(default='', init=False)
    schema_version: str = field(default=VERSION, init=False)

    def __post_init__(self):
        if self.target_scope not in SCOPES or self.priority_band not in PRIORITIES:
            raise ValueError('unsupported target scope or priority')
        if not self.object_ids or not self.cue_ids or not self.source_ids:
            raise ValueError('explicit target references required')
        if self.target_scope == 'object' and len(set(self.object_ids)) != 1:
            raise ValueError('object scope requires one object')
        if (self.target_scope == 'relation') != bool(self.relation_types):
            raise ValueError('relation types required only for relation scope')
        identifier(self.rationale_code, 'rationale_code')
        super().__post_init__()


@dataclass(frozen=True)
class ObservationRequestPlan(_Record):
    scene_id: str
    timestamp: float
    session_id: str
    coordinate_frame_id: str
    source_topological_state_id: str
    source_complex_state_id: str
    source_physical_state_id: str | None
    cues: tuple
    targets: tuple
    uncertainty: tuple
    provenance: Mapping
    plan_id: str = field(default='', init=False)
    schema_version: str = field(default=VERSION, init=False)

    def __post_init__(self):
        number(self.timestamp, 'timestamp', nonnegative=True)
        for name, cls, identity in (('cues', ObservationCue, 'cue_id'), ('targets', ObservationTargetCandidate, 'target_id')):
            values = getattr(self, name)
            if not isinstance(values, (tuple, list)) or any(not isinstance(v, cls) for v in values):
                raise ValueError('typed plan records required')
            if len({getattr(v, identity) for v in values}) != len(values):
                raise ValueError('duplicate plan record')
            key = (lambda v: (PRIORITIES.index(v.priority_band), v.target_id)) if name == 'targets' else (lambda v: v.cue_id)
            object.__setattr__(self, name, tuple(sorted(values, key=key)))
        cues = {c.cue_id: c for c in self.cues}
        for target in self.targets:
            if not set(target.cue_ids) <= cues.keys():
                raise ValueError('unknown cue reference')
            if any(set(cues[i].object_ids) != set(target.object_ids) for i in target.cue_ids):
                raise ValueError('target exceeds cue object scope')
            if set(target.source_ids) != {s for i in target.cue_ids for s in cues[i].source_ids}:
                raise ValueError('inconsistent target source lineage')
        super().__post_init__()


def _check(record, version):
    if record.schema_version != version or replace(record).to_dict() != record.to_dict():
        raise ValueError('invalid source schema, identity or structure')


def _aligned(source, authority):
    for name in ('scene_id', 'timestamp', 'session_id', 'coordinate_frame_id'):
        if getattr(source, name) != getattr(authority, name):
            raise ValueError('current source alignment mismatch: ' + name)


def _validate(topology, complex_state, physical):
    if not isinstance(topology, TopologicalWorldState):
        raise ValueError('TopologicalWorldState required')
    _check(topology, 'topological-world-v0.1')
    tr = topology.transition
    _check(tr, 'topological-world-v0.1')
    if (tr.current_complex_state_id != topology.source_complex_state_id or tr.current_timestamp != topology.timestamp
            or tr.session_id != topology.session_id or tr.coordinate_frame_id != topology.coordinate_frame_id):
        raise ValueError('inconsistent transition lineage')
    if topology.provenance.get('current_scene_id', topology.scene_id) != topology.scene_id:
        raise ValueError('inconsistent current scene lineage')
    for records in (tr.node_states, tr.edge_states, tr.component_correspondences):
        for record in records:
            _check(record, 'topological-world-v0.1')
    nodes = {n.object_id for n in tr.node_states}
    edge_ids = {e.topology_edge_id for e in tr.edge_states}
    for e in tr.edge_states:
        if not set(e.object_ids) <= nodes or not (*e.previous_interaction_ids, *e.current_interaction_ids):
            raise ValueError('invalid topology object/source references')
        for side in ('previous', 'current'):
            sources = e.provenance.get(side + '_sources')
            if sources is not None:
                if {r['interaction_id'] for r in sources} != set(getattr(e, side + '_interaction_ids')):
                    raise ValueError('edge provenance source IDs disagree')
                if any(tuple(r['object_ids']) != e.object_ids or r['interaction_type'] != e.interaction_type
                       or r['status'] not in getattr(e, side + '_epistemic_statuses') for r in sources):
                    raise ValueError('edge provenance semantics disagree')
    for c in tr.component_correspondences:
        if not set(c.object_ids) <= nodes or not set(c.source_edge_ids) <= edge_ids:
            raise ValueError('invalid component references')
    for current, full, identity in ((topology.nodes, tr.node_states, 'node_state_id'),
                                   (topology.edges, tr.edge_states, 'topology_edge_id'),
                                   (topology.components, tr.component_correspondences, 'correspondence_id')):
        lookup = {getattr(r, identity): r.to_dict() for r in full}
        if any(lookup.get(getattr(r, identity)) != r.to_dict() for r in current):
            raise ValueError('current projection conflicts with transition')
    expected_edges = {e.topology_edge_id for e in tr.edge_states
                      if set(e.current_epistemic_statuses).intersection(('present_candidate', 'possible', 'indeterminate'))}
    if expected_edges != {e.topology_edge_id for e in topology.edges}:
        raise ValueError('incomplete current edge projection')
    if {n.object_id for n in topology.nodes} != {o for e in topology.edges for o in e.object_ids}:
        raise ValueError('inconsistent current node projection')
    if {c.correspondence_id for c in topology.components} != {c.correspondence_id for c in tr.component_correspondences if c.current_subsystem_ids}:
        raise ValueError('incomplete current component projection')
    if complex_state is not None:
        if not isinstance(complex_state, ComplexSystemsState):
            raise ValueError('ComplexSystemsState required')
        validate_complex(complex_state)
        _aligned(complex_state, topology)
        if complex_state.complex_state_id != topology.source_complex_state_id:
            raise ValueError('complex source ID mismatch')
        source_edges = {e.interaction_id: e for e in complex_state.interactions}
        if set(source_edges) != {i for e in tr.edge_states for i in e.current_interaction_ids}:
            raise ValueError('incomplete current interaction lineage')
        for e in tr.edge_states:
            for sid in e.current_interaction_ids:
                source = source_edges.get(sid)
                if source is None or source.object_ids != e.object_ids or source.interaction_type != e.interaction_type or source.status not in e.current_epistemic_statuses:
                    raise ValueError('topological interaction source mismatch')
        subsystems = {c.subsystem_id: c for c in complex_state.subsystems}
        for c in tr.component_correspondences:
            if not set(c.current_subsystem_ids) <= subsystems.keys():
                raise ValueError('topological subsystem source mismatch')
    if physical is not None:
        if complex_state is None or not isinstance(physical, LatentPhysicalState):
            raise ValueError('physical support requires matching complex and latent states')
        _check(physical, 'latent-physical-state-0.1')
        _aligned(physical, topology)
        if physical.latent_state_id != complex_state.source_physical_state_id:
            raise ValueError('physical source ID mismatch')
        ids = {o['physical_object_id'] for o in physical.objects}
        for oid in ids:
            identifier(oid, 'physical_object_id')
        for record in (*physical.dynamics, *physical.active_constraints):
            if not set(record.get('object_ids', ())) <= ids:
                raise ValueError('physical reference outside current objects')
        if any(not set(e.object_ids) <= ids for e in complex_state.interactions):
            raise ValueError('complex object references outside physical state')
        snapshot = complex_state.provenance.get('source_snapshots', {}).get('physical_state')
        if snapshot is not None:
            def canonical(value):
                value = bounded_plain(value)
                for name in ('objects', 'relations', 'dynamics', 'active_constraints'):
                    value[name] = sorted({json.dumps(r, sort_keys=True, separators=(',', ':')) for r in value[name]})
                return value
            if canonical(snapshot) != canonical(physical):
                raise ValueError('physical content disagrees with exposed source snapshot')


class ActivePerceptionCore:
    def build(self, topological_state, complex_systems_state=None, physical_state=None):
        t, cs, p = topological_state, complex_systems_state, physical_state
        _validate(t, cs, p)
        cues = {}
        groups = {}

        def add(kind, objects, sources, status, uncertainty, layer, scope, relations=(), extra=None):
            objects, sources, relations = strings(objects), strings(sources), strings(relations)
            if not objects:
                return
            provenance = dict(source_layer=layer, source_record_ids=sources, source_status=status,
                source_object_ids=objects, source_uncertainty=strings(uncertainty),
                epistemic_limitation='observation_need_not_physical_fact', **(extra or {}))
            cue = ObservationCue(kind, objects, sources, status, strings((*uncertainty, *LIMITS)), provenance)
            cues[cue.cue_id] = cue
            groups.setdefault((scope, objects, relations), set()).add(cue.cue_id)

        for edge in t.transition.edge_states:
            sources = (edge.topology_edge_id, *edge.previous_interaction_ids, *edge.current_interaction_ids)
            extra = dict(previous_epistemic_statuses=edge.previous_epistemic_statuses,
                         current_epistemic_statuses=edge.current_epistemic_statuses)
            if edge.status == 'indeterminate' or 'indeterminate' in edge.current_epistemic_statuses:
                add('indeterminate_relation', edge.object_ids, sources, edge.status, edge.uncertainty,
                    'topological', 'relation', (edge.interaction_type,), extra)
            if edge.status in ('appeared', 'disappeared', 'indeterminate'):
                add('relation_change_ambiguity', edge.object_ids, sources, edge.status, edge.uncertainty,
                    'topological', 'relation', (edge.interaction_type,), extra)
        for c in t.transition.component_correspondences:
            if c.status == 'indeterminate' or c.correspondence_type in ('reconfigured_component', 'split_candidate', 'merge_candidate'):
                related = tuple(e.topology_edge_id for e in t.transition.edge_states
                                if e.topology_edge_id in c.source_edge_ids and e.status == 'indeterminate')
                add('component_reconfiguration', c.object_ids,
                    (c.correspondence_id, *c.previous_subsystem_ids, *c.current_subsystem_ids, *c.source_edge_ids),
                    c.status, c.uncertainty, 'topological', 'subsystem', extra=dict(
                        correspondence_type=c.correspondence_type, directly_related_indeterminate_edge_ids=related))
        if cs is not None:
            subsystems = {s.subsystem_id: s for s in cs.subsystems}
            for e in cs.interactions:
                if e.status == 'indeterminate':
                    add('indeterminate_relation', e.object_ids, (e.interaction_id,), e.status, e.uncertainty,
                        'complex_systems', 'relation', (e.interaction_type,))
            for s in cs.subsystems:
                if s.status == 'indeterminate':
                    add('subsystem_uncertainty', s.object_ids, (s.subsystem_id,), s.status, s.uncertainty,
                        'complex_systems', 'subsystem')
            for a in cs.stability_assessments:
                if a.status == 'indeterminate':
                    add('stability_uncertainty', subsystems[a.subsystem_id].object_ids,
                        (a.assessment_id, a.subsystem_id, *a.source_refs), a.status, a.uncertainty,
                        'complex_systems', 'subsystem')
            for a in cs.transition_candidates:
                if a.status == 'indeterminate' or a.pattern_type == 'unresolved_transition':
                    add('transition_uncertainty', subsystems[a.subsystem_id].object_ids,
                        (a.transition_id, a.subsystem_id, *a.source_refs), a.status, a.uncertainty,
                        'complex_systems', 'subsystem', extra=dict(pattern_type=a.pattern_type))
        if p is not None:
            for collection in ('objects', 'relations', 'dynamics'):
                for r in getattr(p, collection):
                    uncertainty = strings(r.get('uncertainty', ()))
                    discontinuity = r.get('signal') == 'tracking_or_motion_discontinuity' and r.get('status') in ('observed', 'estimated', 'possible', 'indeterminate', 'unknown')
                    if not uncertainty and not discontinuity:
                        continue
                    if collection == 'objects':
                        objects = (r['physical_object_id'],)
                    elif collection == 'relations':
                        objects = (r['subject_id'], r['object_id'])
                    else:
                        objects = r.get('object_ids', ())
                    # Mapping records have no record ID: a canonical reference fingerprints symbolic fields only.
                    symbolic = dict(collection=collection, object_ids=strings(objects), status=r.get('status', 'uncertainty_attached'),
                                    signal=r.get('signal'), uncertainty=uncertainty, derived_from=strings(r.get('derived_from', ())))
                    reference = stable_id('physical_cue_source', p.latent_state_id, symbolic)
                    add('physical_state_uncertainty', objects, (p.latent_state_id, reference, *symbolic['derived_from']),
                        symbolic['status'], uncertainty, 'latent_physical', 'object' if len(set(objects)) == 1 else 'object_set',
                        extra=dict(source_collection=collection, source_signal=symbolic['signal'],
                                   source_reference_kind='symbolic_content_reference', discontinuity=discontinuity))
        targets = []
        for (scope, objects, relations), cue_ids in sorted(groups.items()):
            selected = [cues[i] for i in sorted(cue_ids)]
            band = self._priority(selected)
            rationale = ('may_reduce_relation_uncertainty' if scope == 'relation' else
                         'may_reduce_component_ambiguity' if scope == 'subsystem' else 'may_reduce_physical_state_uncertainty')
            targets.append(ObservationTargetCandidate(scope, objects, relations, tuple(cue_ids),
                tuple(sorted({s for c in selected for s in c.source_ids})), rationale, band,
                tuple(sorted({u for c in selected for u in c.uncertainty})),
                dict(epistemic_limitation='target_not_truth_and_priority_not_world_probability',
                     priority_policy='categorical_v01_no_independence_assumption', expected_benefit_measured=False)))
        uncertainty = set(LIMITS) | set(t.uncertainty)
        for source in (cs, p):
            if source is not None:
                uncertainty.update(source.uncertainty)
        uncertainty.update(u for c in cues.values() for u in c.uncertainty)
        provenance = dict(POLICY, complex_support='available' if cs is not None else 'not_supplied',
                          physical_support='available' if p is not None else 'not_supplied',
                          source_uncertainty=dict(topological=t.uncertainty,
                              complex_systems=cs.uncertainty if cs else (), physical=p.uncertainty if p else ()))
        return ObservationRequestPlan(t.scene_id, t.timestamp, t.session_id, t.coordinate_frame_id,
            t.topological_state_id, t.source_complex_state_id, cs.source_physical_state_id if cs else None,
            tuple(cues.values()), tuple(targets), tuple(sorted(uncertainty)), provenance)

    @staticmethod
    def _priority(cues):
        # Two labels from one edge, or copied layers, are not independent evidence.
        if any(c.cue_type == 'component_reconfiguration'
               and (c.epistemic_status == 'indeterminate' or c.provenance.get('correspondence_type') == 'reconfigured_component')
               and c.provenance.get('directly_related_indeterminate_edge_ids') for c in cues):
            return 'high'
        if any(c.cue_type in ('indeterminate_relation', 'component_reconfiguration', 'subsystem_uncertainty',
                             'stability_uncertainty', 'transition_uncertainty') for c in cues):
            return 'medium'
        if all(c.cue_type in ('relation_change_ambiguity', 'physical_state_uncertainty') for c in cues):
            return 'low'
        return 'indeterminate'
