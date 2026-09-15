"""Two-snapshot descriptive topology over explicit Step 24 candidates only."""
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
import json

from ._structured import freeze, identifier, number
from .common_evidence_state import bounded_plain
from .complex_systems import ComplexSystemsState, VERSION as SOURCE_VERSION, VALID
from .evidence import stable_id

VERSION = 'topological-world-v0.1'
LIMITS = ('representation_change_not_event_occurrence', 'topology_not_causality',
          'two_snapshot_comparison_only', 'missing_structure_not_physical_absence')


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
            if isinstance(value, Mapping):
                object.__setattr__(self, f.name, freeze(bounded_plain(value)))
            elif f.name.endswith('_ids') or f.name in ('uncertainty', 'previous_epistemic_statuses', 'current_epistemic_statuses'):
                object.__setattr__(self, f.name, _strings(value))
            elif f.init and f.name.endswith('_id'):
                identifier(value, f.name)
            elif 'timestamp' in f.name:
                number(value, f.name, nonnegative=True)
        if hasattr(self, 'status') and self.status not in ('persistent', 'appeared', 'disappeared', 'indeterminate', 'candidate'):
            raise ValueError('invalid topology status')
        identity = next(f.name for f in fields(self) if not f.init and f.name.endswith('_id'))
        data = self.to_dict()
        data.pop(identity)
        object.__setattr__(self, identity, stable_id(identity, VERSION, data))


@dataclass(frozen=True)
class TopologicalNodeState(_Record):
    object_id: str
    previous_presence: str
    current_presence: str
    status: str
    source_interaction_ids: tuple
    uncertainty: tuple
    provenance: Mapping
    node_state_id: str = field(default='', init=False)
    schema_version: str = field(default=VERSION, init=False)

    def __post_init__(self):
        if any(p not in ('represented', 'not_represented', 'indeterminate') for p in (self.previous_presence, self.current_presence)):
            raise ValueError('invalid presence')
        super().__post_init__()


@dataclass(frozen=True)
class TopologicalEdgeState(_Record):
    object_ids: tuple
    interaction_type: str
    previous_interaction_ids: tuple
    current_interaction_ids: tuple
    status: str
    previous_epistemic_statuses: tuple
    current_epistemic_statuses: tuple
    uncertainty: tuple
    provenance: Mapping
    topology_edge_id: str = field(default='', init=False)
    schema_version: str = field(default=VERSION, init=False)

    def __post_init__(self):
        if len(set(self.object_ids)) < 2:
            raise ValueError('edge requires at least two explicit objects')
        identifier(self.interaction_type, 'interaction_type')
        super().__post_init__()


@dataclass(frozen=True)
class ComponentCorrespondence(_Record):
    previous_subsystem_ids: tuple
    current_subsystem_ids: tuple
    object_ids: tuple
    correspondence_type: str
    status: str
    source_edge_ids: tuple
    uncertainty: tuple
    provenance: Mapping
    correspondence_id: str = field(default='', init=False)
    schema_version: str = field(default=VERSION, init=False)

    def __post_init__(self):
        if self.correspondence_type not in ('persistent_component', 'expanded_component', 'contracted_component',
                'merge_candidate', 'split_candidate', 'appeared_component', 'disappeared_component', 'reconfigured_component'):
            raise ValueError('invalid component correspondence')
        super().__post_init__()


@dataclass(frozen=True)
class TopologicalTransition(_Record):
    previous_complex_state_id: str
    current_complex_state_id: str
    session_id: str
    coordinate_frame_id: str
    previous_timestamp: float
    current_timestamp: float
    node_states: tuple
    edge_states: tuple
    component_correspondences: tuple
    uncertainty: tuple
    provenance: Mapping
    transition_id: str = field(default='', init=False)
    schema_version: str = field(default=VERSION, init=False)

    def __post_init__(self):
        _records(self, 'node_states', TopologicalNodeState, 'node_state_id')
        _records(self, 'edge_states', TopologicalEdgeState, 'topology_edge_id')
        _records(self, 'component_correspondences', ComponentCorrespondence, 'correspondence_id')
        super().__post_init__()
        if self.previous_timestamp >= self.current_timestamp:
            raise ValueError('strictly increasing timestamps required')


@dataclass(frozen=True)
class TopologicalWorldState(_Record):
    scene_id: str
    timestamp: float
    session_id: str
    coordinate_frame_id: str
    source_complex_state_id: str
    nodes: tuple
    edges: tuple
    components: tuple
    transition: TopologicalTransition
    uncertainty: tuple
    provenance: Mapping
    topological_state_id: str = field(default='', init=False)
    schema_version: str = field(default=VERSION, init=False)

    def __post_init__(self):
        _records(self, 'nodes', TopologicalNodeState, 'node_state_id')
        _records(self, 'edges', TopologicalEdgeState, 'topology_edge_id')
        _records(self, 'components', ComponentCorrespondence, 'correspondence_id')
        if not isinstance(self.transition, TopologicalTransition):
            raise ValueError('typed transition required')
        super().__post_init__()


def _records(record, name, cls, identity):
    values = getattr(record, name)
    if not isinstance(values, (tuple, list)) or any(not isinstance(v, cls) for v in values):
        raise ValueError('typed records required')
    if len({getattr(v, identity) for v in values}) != len(values):
        raise ValueError('duplicate topology record')
    object.__setattr__(record, name, tuple(sorted(values, key=lambda v: getattr(v, identity))))


def _key(edge):
    return tuple(sorted(edge.object_ids)), edge.interaction_type


def _validate(state):
    if not isinstance(state, ComplexSystemsState) or state.schema_version != SOURCE_VERSION:
        raise ValueError('Step 24 ComplexSystemsState required')
    number(state.timestamp, 'timestamp', nonnegative=True)
    # Reconstruct only to validate: never silently repair a supplied source.
    if replace(state).to_dict() != state.to_dict():
        raise ValueError('inconsistent source state identity or structure')
    grouped = {}
    for collection in (state.interactions, state.subsystems, state.stability_assessments, state.transition_candidates):
        for item in collection:
            if item.schema_version != SOURCE_VERSION or replace(item).to_dict() != item.to_dict():
                raise ValueError('inconsistent source record')
            for name in ('scene_id', 'timestamp', 'session_id', 'coordinate_frame_id'):
                if hasattr(item, name) and getattr(item, name) != getattr(state, name):
                    raise ValueError('source child context mismatch')
    for edge in state.interactions:
        grouped.setdefault(_key(edge), []).append(edge)
    for edges in grouped.values():
        # Step 24 permits parallel statuses; contradictory duplicates cannot be collapsed.
        if len({e.status for e in edges}) > 1:
            raise ValueError('conflicting duplicate semantic edges')
    edges_by_id = {e.interaction_id: e for e in state.interactions}
    occupied = set()
    for component in state.subsystems:
        selected = [edges_by_id[i] for i in component.interaction_ids]
        objects = set(component.object_ids)
        if any(e.status not in VALID for e in selected) or objects != set().union(*(set(e.object_ids) for e in selected)):
            raise ValueError('component contains absent objects or invalid edges')
        reached = {min(objects)}
        while True:
            expanded = reached | set().union(*(set(e.object_ids) for e in selected if reached.intersection(e.object_ids)))
            if expanded == reached:
                break
            reached = expanded
        if reached != objects or occupied.intersection(objects):
            raise ValueError('disconnected or overlapping source components')
        occupied.update(objects)
    return grouped


def _presence(edges):
    if not edges:
        return 'not_represented'
    return 'represented' if all(e.status in ('possible', 'present_candidate') for e in edges) else 'indeterminate'


def _status(previous, current):
    if 'indeterminate' in (previous, current):
        return 'indeterminate'
    return 'persistent' if previous == current else ('appeared' if current == 'represented' else 'disappeared')


def _uncertainty(*records):
    return tuple(sorted(set(LIMITS).union(*(set(r.uncertainty) for r in records))))


class TopologicalWorldBuilder:
    def build(self, previous_state, current_state):
        previous, current = previous_state, current_state
        before, after = _validate(previous), _validate(current)
        if previous.session_id != current.session_id or previous.coordinate_frame_id != current.coordinate_frame_id:
            raise ValueError('session/frame mismatch')
        if previous.timestamp >= current.timestamp:
            raise ValueError('strictly increasing timestamps required')
        provenance = dict(previous_scene_id=previous.scene_id, current_scene_id=current.scene_id,
            previous_uncertainty=previous.uncertainty, current_uncertainty=current.uncertainty,
            scene_semantics='snapshot_labels_within_same_session_and_frame',
            causal_inference=False, hidden_actor_inference=False, branch_selection=False,
            bayesian_update=False, experience_learning_mutation=False)
        all_sources = (previous, current, *previous.interactions, *current.interactions,
                       *previous.subsystems, *current.subsystems,
                       *previous.stability_assessments, *current.stability_assessments,
                       *previous.transition_candidates, *current.transition_candidates)
        uncertainty = _uncertainty(*all_sources)
        edges = []
        for key in sorted(before.keys() | after.keys()):
            a, b = before.get(key, ()), after.get(key, ())
            status = _status(_presence(a), _presence(b))
            edges.append(TopologicalEdgeState(key[0], key[1], tuple(e.interaction_id for e in a),
                tuple(e.interaction_id for e in b), status, tuple(e.status for e in a), tuple(e.status for e in b),
                _uncertainty(previous, current, *a, *b) + (('source_relation_indeterminate',) if status == 'indeterminate' else ()),
                dict(previous_sources=[e.to_dict() for e in a], current_sources=[e.to_dict() for e in b],
                     semantic_key=key, undirected=True, hyperedge_preserved=True)))
        nodes = []
        objects = sorted({o for e in edges for o in e.object_ids})
        for object_id in objects:
            a = [e for e in previous.interactions if object_id in e.object_ids]
            b = [e for e in current.interactions if object_id in e.object_ids]
            nodes.append(TopologicalNodeState(object_id, _presence(a), _presence(b),
                _status(_presence(a), _presence(b)), tuple(e.interaction_id for e in (*a, *b)),
                _uncertainty(previous, current, *a, *b),
                dict(identity_rule='exact_explicit_object_id', previous_interaction_ids=[e.interaction_id for e in a],
                     current_interaction_ids=[e.interaction_id for e in b])))
        components = self._components(previous, current, edges)
        uncertainty = tuple(sorted(set(uncertainty).union(*(set(c.uncertainty) for c in components),
                                                        *(set(e.uncertainty) for e in edges))))
        transition = TopologicalTransition(previous.complex_state_id, current.complex_state_id,
            current.session_id, current.coordinate_frame_id, previous.timestamp, current.timestamp,
            tuple(nodes), tuple(edges), components, uncertainty, provenance)
        # Unsupported/unavailable relations remain transition diagnostics, not graph edges.
        current_objects = {o for e in current.interactions if e.status in VALID for o in e.object_ids}
        # World arrays are the current projection; transition retains disappeared records.
        return TopologicalWorldState(current.scene_id, current.timestamp, current.session_id,
            current.coordinate_frame_id, current.complex_state_id,
            tuple(n for n in nodes if n.object_id in current_objects),
            tuple(e for e in edges if any(s in VALID for s in e.current_epistemic_statuses)),
            tuple(c for c in components if c.current_subsystem_ids), transition, uncertainty, provenance)

    @staticmethod
    def _components(previous, current, edges):
        vertices = {(0, c.subsystem_id): c for c in previous.subsystems}
        vertices.update({(1, c.subsystem_id): c for c in current.subsystems})
        adjacency = {k: set() for k in vertices}
        for a, ca in vertices.items():
            for b, cb in vertices.items():
                if a[0] != b[0] and set(ca.object_ids).intersection(cb.object_ids):
                    adjacency[a].add(b)
        remaining = set(vertices)
        result = []
        while remaining:
            group, pending = set(), {min(remaining)}
            while pending:
                k = pending.pop()
                if k in group:
                    continue
                group.add(k)
                pending.update(adjacency[k] - group)
            remaining -= group
            a = [vertices[k] for k in sorted(group) if k[0] == 0]
            b = [vertices[k] for k in sorted(group) if k[0] == 1]
            ao = set().union(*(set(c.object_ids) for c in a))
            bo = set().union(*(set(c.object_ids) for c in b))
            if not a:
                kind = 'appeared_component'
            elif not b:
                kind = 'disappeared_component'
            elif len(a) == len(b) == 1:
                kind = ('persistent_component' if ao == bo else 'expanded_component' if ao < bo
                        else 'contracted_component' if bo < ao else 'reconfigured_component')
            elif len(a) == 1:
                kind = 'split_candidate'
            elif len(b) == 1:
                kind = 'merge_candidate'
            else:
                kind = 'reconfigured_component'
            source_ids = {i for c in (*a, *b) for i in c.interaction_ids}
            source_edges = [e for e in edges if source_ids.intersection((*e.previous_interaction_ids, *e.current_interaction_ids))]
            ambiguous = kind == 'reconfigured_component' or any(c.status == 'indeterminate' for c in (*a, *b)) or any(e.status == 'indeterminate' for e in source_edges)
            # An unavailable opposite-snapshot relation cannot establish loss or gain.
            if any(e.status == 'indeterminate' and set(e.object_ids).intersection(ao | bo) for e in edges):
                ambiguous = True
            result.append(ComponentCorrespondence(tuple(c.subsystem_id for c in a), tuple(c.subsystem_id for c in b),
                tuple(ao | bo), kind, 'indeterminate' if ambiguous else 'candidate',
                tuple(e.topology_edge_id for e in source_edges),
                _uncertainty(previous, current, *a, *b, *source_edges) + (('ambiguous_component_correspondence',) if ambiguous else ()),
                dict(algorithm='bipartite_object_set_overlap_connected_groups',
                     previous_components=[c.to_dict() for c in a], current_components=[c.to_dict() for c in b])))
        return tuple(result)
