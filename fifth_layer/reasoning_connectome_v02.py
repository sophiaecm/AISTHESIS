"""Multi-layer routing only; no truth fusion, belief update or application action."""
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from math import fsum
import json

from .orchestration import OrchestrationContext, ReasonerRegistry, ReasonerOrchestrator
from .world_model._structured import freeze, identifier, number
from .world_model.evidence import stable_id
from .world_model.common_evidence_state import CommonEvidenceState, bounded_plain
from .world_model.hybrid_world_state import _alignment
from .world_model.latent_physical_state import LatentPhysicalState
from .world_model.physics_constraints import PhysicsConstraintBundle, PhysicsTransitionAssessment
from .world_model.cross_modal_consequences import CrossModalConsequenceBundle
from .world_model.multiple_futures import MultipleFutureBundle
from .world_model.bayesian_belief_state import BayesianBeliefState
from .world_model.experience_learning_v02 import ExperienceLearningViewV02

VERSION = 'reasoning-connectome-v0.2'
LAYERS = ('evidence', 'physical', 'physics_constraints', 'cross_modal', 'multiple_futures', 'belief', 'experience')
_OPTIONAL = {
    'physical': ('physical_state', LatentPhysicalState, 'latent-physical-state-0.1'),
    'physics_constraints': ('physics_constraints', (PhysicsConstraintBundle, PhysicsTransitionAssessment), None),
    'cross_modal': ('cross_modal_consequences', CrossModalConsequenceBundle, 'cross-modal-consequences-0.2'),
    'multiple_futures': ('multiple_futures', MultipleFutureBundle, 'multiple-futures-0.2'),
    'belief': ('belief_state', BayesianBeliefState, 'bayesian-belief-state-v0.1'),
    'experience': ('experience_view', ExperienceLearningViewV02, 'experience-learning-v0.2'),
}
POLICY = dict(relevance_is_probability=False, truth_decision='not_performed',
    Bayesian_feedback=False, branch_selected=False, current_observation=False,
    experience_learning=False, physical_action=False, calibrated=False,
    selected_means_correct=False, excluded_means_false=False)


def _frozen(value):
    return freeze(bounded_plain(value))


class _Serializable:
    def to_dict(self):
        return bounded_plain({f.name: getattr(self, f.name) for f in fields(self) if not f.name.startswith('_')})

    def to_json(self):
        return json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'), allow_nan=False)


def _strings(values):
    if not isinstance(values, (tuple, list)):
        raise ValueError('ordered string sequence required')
    for value in values:
        identifier(value, 'routing category')
    return tuple(sorted(set(values)))


def _current_alignment(value, common):
    for name in ('scene_id', 'session_id', 'timestamp', 'coordinate_frame_id'):
        if hasattr(value, name) and getattr(value, name) != getattr(common, name):
            raise ValueError(f'current layer {name} mismatch')


def _experience_alignment(view, common, belief):
    # Historical scenes/frames are not current context. Validate current headers
    # separately from historical records; never run current-frame checks on history.
    identifier(view.source_belief_state_id, 'source_belief_state_id')
    queries = {q.hypothesis_id: q for q in view.queries}
    rows = {r.hypothesis_id: r for r in view.rows}
    if len(queries) != len(view.queries) or set(queries) != set(rows):
        raise ValueError('experience query/row mismatch')
    for q in view.queries:
        _current_alignment(q, common)
        if q.belief_state_id != view.source_belief_state_id or replace(q) != q:
            raise ValueError('experience query belief or identity mismatch')
        if q.provenance.get('coordinate_frame_id') != common.coordinate_frame_id:
            raise ValueError('experience query coordinate_frame_id mismatch')
        _alignment(q.provenance, common)
    for row in view.rows:
        if replace(row) != row or row.source_future_id != queries[row.hypothesis_id].source_future_id:
            raise ValueError('experience row mismatch')
        if not view.enabled and (row.historical_contribution != 0 or row.retrieved_experiences):
            raise ValueError('disabled experience cannot contain active history')
        for r in row.retrieved_experiences:
            if r.query_id != queries[row.hypothesis_id].query_id or replace(r) != r:
                raise ValueError('retrieval query or identity mismatch')
            for stamp in r.historical_timestamps.values():
                number(stamp, 'historical timestamp', nonnegative=True)
            if common.timestamp is None or any(t >= common.timestamp for t in r.historical_timestamps.values()):
                raise ValueError('experience must be strictly historical')
            if r.provenance.get('historical_prediction_provenance', {}).get('session_id') != common.session_id:
                raise ValueError('historical session mismatch')
    if belief is not None:
        if view.source_belief_state_id != belief.belief_state_id or set(rows) != set(belief.source_future_ids):
            raise ValueError('experience view must match supplied belief')
        for b in belief.beliefs:
            if rows[b.hypothesis_id].base_posterior_probability != b.posterior_probability:
                raise ValueError('experience base posterior mismatch')


@dataclass(frozen=True)
class MultiLayerReasoningContext(_Serializable):
    orchestration_context: OrchestrationContext
    physical_state: LatentPhysicalState | None = None
    physics_constraints: PhysicsConstraintBundle | PhysicsTransitionAssessment | None = None
    cross_modal_consequences: CrossModalConsequenceBundle | None = None
    multiple_futures: MultipleFutureBundle | None = None
    belief_state: BayesianBeliefState | None = None
    experience_view: ExperienceLearningViewV02 | None = None
    descriptors: tuple = field(init=False)
    layer_availability: Mapping = field(init=False)
    uncertainty: Mapping = field(init=False)
    source_references: Mapping = field(init=False)
    context_id: str = field(init=False)
    schema_version: str = field(default=VERSION, init=False)

    def __post_init__(self):
        if not isinstance(self.orchestration_context, OrchestrationContext):
            raise ValueError('OrchestrationContext with CommonEvidenceState required')
        common = self.orchestration_context.common_evidence_state
        if not isinstance(common, CommonEvidenceState) or common.schema_version != 'common-evidence-state-0.1':
            raise ValueError('CommonEvidenceState v0.1 required')
        if replace(self.orchestration_context) != self.orchestration_context or replace(common) != common:
            raise ValueError('inconsistent orchestration context')
        refs = {'evidence': dict(reference=self.orchestration_context.execution_id,
                                schema_version=common.schema_version, sources=common.source_references)}
        availability = {'evidence': 'available'}
        source_uncertainty = {'evidence': common.uncertainty}
        flags = set(common.uncertainty['integration'])
        for layer, (name, cls, version) in _OPTIONAL.items():
            value = getattr(self, name)
            availability[layer] = 'unavailable' if value is None else 'available'
            if value is None:
                refs[layer] = None
                flags.add(layer + '.unavailable')
                continue
            if not isinstance(value, cls):
                raise ValueError(f'invalid {layer} type')
            expected = version or ('physics-transition-0.2' if isinstance(value, PhysicsTransitionAssessment) else 'physics-constraints-0.2')
            if value.schema_version != expected or replace(value) != value:
                raise ValueError(f'invalid {layer} version or structure')
            _current_alignment(value, common)
            if layer != 'experience':
                _alignment(bounded_plain(value), common)
            refs[layer] = dict(reference=stable_id('routing-source', bounded_plain(value)), schema_version=value.schema_version)
            source_uncertainty[layer] = getattr(value, 'uncertainty', ())
            if isinstance(source_uncertainty[layer], (tuple, list)):
                flags.update(layer + '.' + u for u in source_uncertainty[layer])
            if layer == 'physics_constraints':
                bundle = value.constraints if isinstance(value, PhysicsTransitionAssessment) else value
                source_uncertainty[layer] = {r.constraint_id: r.uncertainty for r in bundle.results}
                flags.update(layer + '.' + u for r in bundle.results for u in r.uncertainty)
        if self.experience_view is not None:
            _experience_alignment(self.experience_view, common, self.belief_state)
            if not self.experience_view.enabled:
                availability['experience'] = 'disabled'
                flags.add('experience.disabled')
        if self.belief_state is not None and self.multiple_futures is not None:
            b, f = self.belief_state, self.multiple_futures
            if b.source_multiple_futures_id != stable_id('multiple-futures-source', bounded_plain(f)):
                raise ValueError('belief/future source mismatch')
            if {x.future_id: x for x in f.branches} != {x.source_future_id: x.source_future for x in b.beliefs}:
                raise ValueError('belief source branches mismatch')
        physical_id = self.physical_state.latent_state_id if self.physical_state else None
        branches = list(self.multiple_futures.branches) if self.multiple_futures else []
        if self.belief_state:
            branches.extend(b.source_future for b in self.belief_state.beliefs)
        candidates = self.cross_modal_consequences.candidates if self.cross_modal_consequences else ()
        source_ids = {x.source_physical_state_id for x in (*branches, *candidates)}
        if self.multiple_futures:
            source_ids.add(self.multiple_futures.source_physical_state_id)
        if self.cross_modal_consequences:
            declared = self.cross_modal_consequences.provenance.get('source_physical_state_id')
            if declared is not None:
                source_ids.add(declared)
        if physical_id is not None:
            source_ids.add(physical_id)
        if len(source_ids) > 1:
            raise ValueError('physical source identity mismatch between layers')
        if physical_id is not None:
            if any(x.source_physical_state_id != physical_id for x in (*branches, *candidates)) or (
                    self.multiple_futures and self.multiple_futures.source_physical_state_id != physical_id):
                raise ValueError('physical source identity mismatch')
            declared = self.cross_modal_consequences.provenance.get('source_physical_state_id') if self.cross_modal_consequences else None
            if declared is not None and declared != physical_id:
                raise ValueError('cross-modal physical source mismatch')
        if self.cross_modal_consequences:
            ids = {c.consequence_id for c in candidates}
            if any(not set(b.source_consequence_ids) <= ids for b in branches):
                raise ValueError('future consequence references absent from supplied bundle')
        if self.physics_constraints is not None:
            bundle = self.physics_constraints.constraints if isinstance(self.physics_constraints, PhysicsTransitionAssessment) else self.physics_constraints
            ids = {r.constraint_id for r in bundle.results}
            if any(not set(x.source_constraint_ids) <= ids for x in (*branches, *candidates)):
                raise ValueError('constraint source references absent from supplied bundle')
        descriptors = _descriptors(self)
        for d in descriptors:
            if any(s in d['category'] for s in ('indeterminate', 'unavailable', 'unsupported')):
                flags.add(d['category'])
        object.__setattr__(self, 'descriptors', _frozen(descriptors))
        object.__setattr__(self, 'layer_availability', _frozen(availability))
        object.__setattr__(self, 'source_references', _frozen(refs))
        object.__setattr__(self, 'uncertainty', _frozen(dict(sources=source_uncertainty, flags=tuple(sorted(flags)))))
        object.__setattr__(self, 'context_id', stable_id('multilayer-context', VERSION, refs,
            descriptors, availability, self.uncertainty))

    def to_dict(self):
        # Full immutable sources are retained in memory, represented by content
        # references in serialization to avoid replicating large source trees.
        return bounded_plain(dict(context_id=self.context_id, schema_version=VERSION,
            orchestration_context_reference=self.orchestration_context.execution_id,
            descriptors=self.descriptors, layer_availability=self.layer_availability,
            uncertainty=self.uncertainty, source_references=self.source_references))


def _descriptors(context):
    result = []
    def add(layer, category, *, refs=(), active=True, **details):
        data = dict(layer=layer, category=layer + '.' + category, source_refs=tuple(refs),
                    routable=active, details=details)
        result.append(dict(descriptor_id=stable_id('routing-descriptor', data), **data))
    common = context.orchestration_context.common_evidence_state
    for e in common.evidence_items:
        active = e.source_type.value != 'experience' and e.epistemic_status != 'unavailable'
        for category in ('family.' + e.source_type.value, 'type.' + e.evidence_type,
                         'status.' + (e.epistemic_status or 'unspecified')):
            add('evidence', category, refs=(e.evidence_id,), active=active,
                evidence_id=e.evidence_id, source_family=e.source_type.value,
                evidence_type=e.evidence_type, epistemic_status=e.epistemic_status, timestamp=e.timestamp)
    for layer, (name, _, _) in _OPTIONAL.items():
        value = getattr(context, name)
        if value is None:
            add(layer, 'unavailable', active=False)
        else:
            add(layer, 'present', active=layer != 'experience' or value.enabled)
    p = context.physical_state
    if p:
        for obj in p.objects:
            attr = obj.get('attributes', {}).get('motion_state', {})
            if attr.get('value') is not None and attr.get('status') in ('observed', 'estimated'):
                add('physical', 'motion_state_present', refs=(p.latent_state_id,), status=attr['status'])
        for record in (*p.relations, *p.dynamics):
            if record.get('signal') in ('contact_candidate', 'collision_risk_geometry', 'tracking_or_motion_discontinuity'):
                add('physical', record['signal'] + '_present', refs=(p.latent_state_id,), status=record.get('status'))
        add('physical', 'inventory', active=False, object_count=len(p.objects), active_constraint_count=len(p.active_constraints))
    c = context.physics_constraints
    if c:
        bundle = c.constraints if isinstance(c, PhysicsTransitionAssessment) else c
        for r in bundle.results:
            for category in ('type.' + r.constraint_type, 'status.' + r.status):
                add('physics_constraints', category, refs=(r.constraint_id,), status=r.status)
    c = context.cross_modal_consequences
    if c:
        for candidate in c.candidates:
            for category in ('modality.' + candidate.modality.value, 'status.' + candidate.status.value,
                             'event_family.' + candidate.event_family):
                add('cross_modal', category, refs=(candidate.consequence_id,), candidate_status=candidate.status.value,
                    epistemic_role='consequence_candidate_not_observation')
        add('cross_modal', 'inventory', active=False, candidate_count=len(c.candidates))
    f = context.multiple_futures
    if f:
        for b in f.branches:
            for category in ('family.' + b.branch_family, 'status.' + b.status.value, 'horizon.' + b.horizon_kind.value):
                add('multiple_futures', category, refs=(b.future_id,), epistemic_role='conditional_branch')
        for r in f.branch_relations:
            add('multiple_futures', 'relation.' + r.relation.value, refs=(r.relation_id,))
        add('multiple_futures', 'inventory', active=False, branch_count=len(f.branches))
    b = context.belief_state
    if b:
        for category in ('initialization.' + b.initialization_mode.value, 'update.' + b.update_status.value,
                         'posterior.' + ('available' if b.beliefs and all(x.posterior_probability is not None for x in b.beliefs) else 'unavailable')):
            add('belief', category, refs=(b.belief_state_id,))
        add('belief', 'inventory', active=False, hypothesis_count=len(b.beliefs))
    v = context.experience_view
    if v:
        add('experience', 'enabled' if v.enabled else 'disabled', active=v.enabled, refs=(v.view_id,))
        if v.enabled:
            retrieved = [r for row in v.rows for r in row.retrieved_experiences]
            if retrieved:
                add('experience', 'relevant_history_present', refs=(v.view_id,))
                if not any(row.historical_contribution != 0 for row in v.rows):
                    add('experience', 'neutral_only', refs=(v.view_id,))
            for category, present in (
                ('positive_context_present', any(r.historical_contribution > 0 for r in v.rows)),
                ('negative_context_present', any(r.historical_contribution < 0 for r in v.rows)),
                ('suppressed_rows_present', any(r.provenance.get('suppression_reason') for r in v.rows)),
                ('contribution_available', any(r.base_posterior_probability is not None for r in v.rows))):
                if present:
                    add('experience', category, refs=(v.view_id,))
            add('experience', 'inventory', active=False, retrieval_count=len(retrieved))
    # Identical categorical physical descriptors need not repeat.
    return tuple(sorted({d['descriptor_id']: d for d in result}.values(), key=lambda d: (d['layer'], d['category'], d['descriptor_id'])))


@dataclass(frozen=True)
class ReasonerRoutingProfile(_Serializable):
    reasoner_id: str
    preferred_layers: tuple = ()
    preferred_descriptor_categories: tuple = ()
    optional_descriptor_categories: tuple = ()
    blocked_when_flags: tuple = ()
    routing_role: str = 'caller_defined'
    provenance: Mapping = field(default_factory=dict)

    def __post_init__(self):
        identifier(self.reasoner_id, 'reasoner_id')
        identifier(self.routing_role, 'routing_role')
        for name in ('preferred_layers', 'preferred_descriptor_categories', 'optional_descriptor_categories', 'blocked_when_flags'):
            object.__setattr__(self, name, _strings(getattr(self, name)))
        if not set(self.preferred_layers) <= set(LAYERS):
            raise ValueError('unknown routing layer')
        if not isinstance(self.provenance, Mapping):
            raise ValueError('profile provenance must be a mapping')
        object.__setattr__(self, 'provenance', _frozen(self.provenance))


@dataclass(frozen=True)
class ConnectomeConfigV02(_Serializable):
    min_relevance: float = .3
    max_selected_reasoners: int | None = None
    base_relevance: float = .2
    evidence_layer_increment: float = .2
    physical_layer_increment: float = .1
    physics_constraint_increment: float = .1
    cross_modal_increment: float = .1
    future_layer_increment: float = .1
    belief_layer_increment: float = .1
    experience_layer_increment: float = .05
    descriptor_match_increment: float = .1

    def __post_init__(self):
        for f in fields(self):
            if f.name != 'max_selected_reasoners':
                number(getattr(self, f.name), f.name, unit=True)
        if self.max_selected_reasoners is not None and (type(self.max_selected_reasoners) is not int or self.max_selected_reasoners < 0):
            raise ValueError('selection limit must be nonnegative integer or None')

    def increments(self):
        return dict(zip(LAYERS, (self.evidence_layer_increment, self.physical_layer_increment,
            self.physics_constraint_increment, self.cross_modal_increment, self.future_layer_increment,
            self.belief_layer_increment, self.experience_layer_increment)))


@dataclass(frozen=True)
class ReasoningGraphSnapshotV02(_Serializable):
    session_id: str
    scene_id: str
    timestamp: float | None
    context_reference: str
    registry_reference: str
    profile_reference: str
    nodes: Mapping
    edges: tuple
    relevance_scores: Mapping
    relevance_breakdown: Mapping
    selected_reasoner_ids: tuple
    exclusion_reasons: Mapping
    layer_availability: Mapping
    source_references: Mapping
    uncertainty: Mapping
    descriptors: tuple
    config: ConnectomeConfigV02
    provenance: Mapping = field(default_factory=lambda: _frozen(POLICY))
    _execution_bindings: tuple = field(default=(), repr=False, compare=False)
    snapshot_id: str = field(default='', init=False)
    schema_version: str = field(default=VERSION, init=False)

    def __post_init__(self):
        for name in ('session_id', 'scene_id', 'context_reference', 'registry_reference', 'profile_reference'):
            identifier(getattr(self, name), name)
        if not isinstance(self.config, ConnectomeConfigV02):
            raise ValueError('ConnectomeConfigV02 required')
        if self.timestamp is not None:
            number(self.timestamp, 'timestamp', nonnegative=True)
        object.__setattr__(self, '_execution_bindings', tuple(self._execution_bindings))
        for name in ('nodes', 'relevance_scores', 'relevance_breakdown', 'exclusion_reasons',
                     'layer_availability', 'source_references', 'uncertainty', 'provenance',
                     'edges', 'selected_reasoner_ids', 'descriptors'):
            object.__setattr__(self, name, _frozen(getattr(self, name)))
        ids = set(self.nodes)
        if set(self.relevance_scores) != ids or set(self.relevance_breakdown) != ids:
            raise ValueError('every node requires a score and breakdown')
        selected = self.selected_reasoner_ids
        if len(set(selected)) != len(selected) or not set(selected) <= ids or set(self.exclusion_reasons) != ids - set(selected):
            raise ValueError('invalid selection or exclusion inventory')
        for score in self.relevance_scores.values():
            number(score, 'relevance', unit=True)
        for edge in self.edges:
            if edge['target_reasoner_id'] not in ids or edge['source_context_reference'] != self.context_reference or edge['relation'] != 'context_layer_activates_reasoner':
                raise ValueError('invalid routing edge')
        data = self.to_dict()
        data.pop('snapshot_id')
        object.__setattr__(self, 'snapshot_id', stable_id('routing-snapshot', data))


@dataclass(frozen=True)
class DynamicReasoningConnectomeV02:
    config: ConnectomeConfigV02 = field(default_factory=ConnectomeConfigV02)

    def __post_init__(self):
        if not isinstance(self.config, ConnectomeConfigV02):
            raise ValueError('ConnectomeConfigV02 required')

    def route(self, context, registry, profiles=()):
        if not isinstance(context, MultiLayerReasoningContext) or not isinstance(registry, ReasonerRegistry):
            raise ValueError('MultiLayerReasoningContext and ReasonerRegistry required')
        context = replace(context)  # Validate current sources and recompute content references.
        if not isinstance(profiles, (tuple, list)) or any(not isinstance(p, ReasonerRoutingProfile) for p in profiles):
            raise ValueError('ordered routing profiles required')
        profile_map = {p.reasoner_id: p for p in profiles}
        ids = {r.reasoner_id for r in registry.registrations}
        if len(profile_map) != len(profiles) or not set(profile_map) <= ids:
            raise ValueError('duplicate profile or unregistered reasoner')
        plan = {e.reasoner_id: e for e in ReasonerOrchestrator(registry).plan(context.orchestration_context)}
        active = tuple(d for d in context.descriptors if d['routable'])
        present = {d['layer'] for d in active}
        flags = set(context.uncertainty['flags'])
        increments = self.config.increments()
        nodes, scores, breakdowns, edges = {}, {}, {}, []
        for r in sorted(registry.registrations, key=lambda r: r.reasoner_id):
            identity = r.reasoner_id
            nodes[identity] = dict(node_id=identity, **r.metadata())
            profile = profile_map.get(identity)
            layers = tuple(l for l in LAYERS if profile and l in profile.preferred_layers and l in present)
            categories = set() if profile is None else set(profile.preferred_descriptor_categories + profile.optional_descriptor_categories)
            matches = tuple(d for d in active if d['category'] in categories)
            bonuses = {l: increments[l] if l in layers else 0. for l in LAYERS}
            descriptor_bonus = self.config.descriptor_match_increment if matches else 0.
            raw = fsum((self.config.base_relevance, descriptor_bonus, *bonuses.values()))
            scores[identity] = min(1., max(0., raw))
            blocked = tuple(sorted(flags & set(profile.blocked_when_flags))) if profile else ()
            breakdowns[identity] = dict(base=self.config.base_relevance, layer_bonuses=bonuses,
                descriptor_bonus=descriptor_bonus, matched_categories=tuple(sorted({d['category'] for d in matches})),
                unclipped_relevance=raw, clipping_adjustment=scores[identity] - raw,
                blocked_flags=blocked, eligibility=bounded_plain(plan[identity]),
                profile_policy='explicit' if profile else 'base_only', uncertainty_penalty=0.)
            contributions = [(l, 'preferred_layer_presence', bonuses[l], tuple(d for d in active if d['layer'] == l)) for l in layers]
            if matches:
                # One category bonus, attributed to the first canonical matching
                # descriptor's layer; the breakdown retains every category match.
                contributions.append((matches[0]['layer'], 'descriptor_category_presence', descriptor_bonus, (matches[0],)))
            for layer, rule, increment, ds in contributions:
                if increment:
                    data = dict(source_context_reference=context.context_id, target_reasoner_id=identity,
                        source_layer=layer, relation='context_layer_activates_reasoner', routing_rule=rule,
                        increment=increment, descriptor_refs=tuple(d['descriptor_id'] for d in ds),
                        descriptor_categories=tuple(sorted({d['category'] for d in ds})), truth_semantics=False)
                    edges.append(dict(edge_id=stable_id('routing-edge', data), **data))
        selected, excluded = [], {}
        for identity in sorted(scores, key=lambda i: (-scores[i], i)):
            reason = ('step15_ineligible' if plan[identity].status != 'eligible' else
                      'profile_blocked' if breakdowns[identity]['blocked_flags'] else
                      'below_relevance_threshold' if scores[identity] < self.config.min_relevance else
                      'selection_limit' if self.config.max_selected_reasoners is not None and
                        len(selected) >= self.config.max_selected_reasoners else None)
            if reason:
                excluded[identity] = dict(reason=reason, eligibility=bounded_plain(plan[identity]),
                                         blocked_flags=breakdowns[identity]['blocked_flags'])
            else:
                selected.append(identity)
        common = context.orchestration_context.common_evidence_state
        return ReasoningGraphSnapshotV02(common.session_id, common.scene_id, common.timestamp,
            context.context_id, stable_id('routing-registry', nodes),
            stable_id('routing-profiles', [profile_map[i].to_dict() for i in sorted(profile_map)]),
            nodes, tuple(sorted(edges, key=lambda e: e['edge_id'])), scores, breakdowns, tuple(selected), excluded,
            context.layer_availability, context.source_references, context.uncertainty, context.descriptors, self.config,
            _execution_bindings=tuple((r.reasoner_id, r.execute) for r in sorted(registry.registrations, key=lambda r: r.reasoner_id)))

    def execute(self, context, registry, profiles, snapshot):
        expected = self.route(context, registry, profiles)
        if not isinstance(snapshot, ReasoningGraphSnapshotV02) or snapshot.to_json() != expected.to_json():
            raise ValueError('stale or tampered routing snapshot')
        # Callables have no portable content identity in frozen registrations.
        # Bind in-process execution separately, without nondeterministic JSON IDs.
        if len(snapshot._execution_bindings) != len(expected._execution_bindings) or any(
                a != b or fn is not other for (a, fn), (b, other) in zip(snapshot._execution_bindings, expected._execution_bindings)):
            raise ValueError('registry execution binding changed; route again')
        registrations = {r.reasoner_id: r for r in registry.registrations}
        selected = ReasonerRegistry(tuple(replace(registrations[i], priority=order)
            for order, i in enumerate(expected.selected_reasoner_ids)))
        return ReasonerOrchestrator(selected).run(context.orchestration_context)
