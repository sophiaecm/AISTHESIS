"""Explicit context routing heuristics; relevance is not truth probability."""
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from math import fsum
import json

from .orchestration import OrchestrationContext, ReasonerRegistry, ReasonerOrchestrator
from .world_model._structured import freeze, identifier, number
from .world_model.evidence import stable_id
from .world_model.common_evidence_state import CommonEvidenceState, FAMILIES, bounded_plain


def _frozen(value):
    return freeze(bounded_plain(value))


def _default_matches():
    return {'physics': ('physical', 'physics', 'physics_constraint'),
            'physical': ('physical',), 'temporal': ('temporal',),
            'occlusion': ('occlusion',), 'sensory': ('sensory',),
            'learned_representation': ('learned_representation',),
            'documentary': ('documentary',)}


@dataclass(frozen=True)
class ConnectomeConfig:
    min_relevance: float = .3
    max_selected_reasoners: int | None = None
    base_relevance: float = .2
    evidence_match_increment: float = .4
    category_match_increment: float = .2
    capability_match_increment: float = .1
    family_evidence: Mapping = field(default_factory=_default_matches)

    def __post_init__(self):
        for name in ('min_relevance', 'base_relevance', 'evidence_match_increment',
                     'category_match_increment', 'capability_match_increment'):
            number(getattr(self, name), name, unit=True)
        if self.max_selected_reasoners is not None and (
                type(self.max_selected_reasoners) is not int or self.max_selected_reasoners < 0):
            raise ValueError('max_selected_reasoners must be nonnegative integer or None')
        if not isinstance(self.family_evidence, Mapping):
            raise ValueError('family_evidence must be a mapping')
        matches = {}
        for family, sources in self.family_evidence.items():
            identifier(family, 'reasoner_family')
            if not isinstance(sources, (tuple, list)) or any(s not in FAMILIES or s == 'experience' for s in sources):
                raise ValueError('routing source must be a supported non-experience family')
            matches[family] = tuple(sorted(set(sources)))
        object.__setattr__(self, 'family_evidence', _frozen(matches))


@dataclass(frozen=True)
class ConnectomeContext:
    orchestration_context: OrchestrationContext
    descriptors: tuple = field(init=False)
    uncertainty: Mapping = field(init=False)

    def __post_init__(self):
        if not isinstance(self.orchestration_context, OrchestrationContext):
            raise ValueError('expected OrchestrationContext')
        state = self.orchestration_context.common_evidence_state
        descriptors = tuple({'evidence_id': e.evidence_id, 'source_family': e.source_type.value,
            'category': e.evidence_type, 'epistemic_status': e.epistemic_status,
            'timestamp': e.timestamp} for e in state.evidence_items)
        flags = set()
        if not descriptors:
            flags.add('missing_context')
        if state.uncertainty['integration'] or any(d['epistemic_status'] in (None, 'unknown', 'unavailable') for d in descriptors):
            flags.add('partial_context')
        object.__setattr__(self, 'descriptors', _frozen(descriptors))
        object.__setattr__(self, 'uncertainty', _frozen({'source_uncertainty': state.uncertainty,
                                                     'routing_context': tuple(sorted(flags))}))

    @classmethod
    def from_state(cls, state):
        if isinstance(state, cls):
            return state
        if isinstance(state, CommonEvidenceState):
            state = OrchestrationContext(state)
        return cls(state)


@dataclass(frozen=True)
class ReasoningGraphSnapshot:
    session_id: str
    scene_id: str
    timestamp: float | None
    context_reference: str
    registry_reference: str
    nodes: Mapping
    edges: tuple
    relevance_scores: Mapping
    relevance_breakdown: Mapping
    selected_reasoner_ids: tuple[str, ...]
    exclusion_reasons: Mapping
    config: ConnectomeConfig
    source_evidence_refs: Mapping
    uncertainty: Mapping
    context_descriptors: tuple
    schema_version: str = field(default='reasoning-connectome-0.1', init=False)

    def __post_init__(self):
        for name in ('session_id', 'scene_id', 'context_reference', 'registry_reference'):
            identifier(getattr(self, name), name)
        if self.timestamp is not None:
            number(self.timestamp, 'timestamp', nonnegative=True)
        if not isinstance(self.config, ConnectomeConfig):
            raise ValueError('expected ConnectomeConfig')
        for name in ('nodes', 'relevance_scores', 'relevance_breakdown', 'exclusion_reasons',
                     'source_evidence_refs', 'uncertainty'):
            if not isinstance(getattr(self, name), Mapping):
                raise ValueError('snapshot metadata must be mappings')
            object.__setattr__(self, name, _frozen(getattr(self, name)))
        for name in ('edges', 'context_descriptors', 'selected_reasoner_ids'):
            if not isinstance(getattr(self, name), (tuple, list)):
                raise ValueError('snapshot sequences must be ordered')
            object.__setattr__(self, name, _frozen(getattr(self, name)))
        ids = set(self.nodes)
        selected = self.selected_reasoner_ids
        if len(set(selected)) != len(selected) or not set(selected) <= ids:
            raise ValueError('duplicate or unknown selected reasoner ID')
        if set(self.relevance_scores) != ids or set(self.relevance_breakdown) != ids:
            raise ValueError('scores must describe every node')
        if set(self.exclusion_reasons) != ids - set(selected):
            raise ValueError('each non-selected node requires an exclusion reason')
        for identity, node in self.nodes.items():
            if node.get('node_id') != identity or node.get('reasoner_id') != identity:
                raise ValueError('node ID must reference the existing reasoner ID')
            number(self.relevance_scores[identity], 'relevance', unit=True)
        for edge in self.edges:
            if edge['target_reasoner_id'] not in ids or edge['source_context_reference'] != self.context_reference:
                raise ValueError('edge endpoints must match snapshot')
            if edge['relation'] != 'context_activates_reasoner':
                raise ValueError('unsupported routing edge relation')
        bounded_plain(self)

    @property
    def excluded_reasoner_ids(self):
        return tuple(sorted(self.exclusion_reasons))

    def to_dict(self):
        return bounded_plain(self)

    def to_json(self):
        return json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'), allow_nan=False)


def _score(registration, context, config):
    """Presence bonuses, never evidence counts or source confidence/values."""
    descriptors = tuple(d for d in context.descriptors
                        if d['epistemic_status'] != 'unavailable' and d['source_family'] != 'experience')
    families = (set(config.family_evidence.get(registration.reasoner_family, ())) |
                set(registration.required_evidence) | set(registration.optional_evidence)) - {'experience'}
    family_refs = tuple(d['evidence_id'] for d in descriptors if d['source_family'] in families)
    category_refs = tuple(d['evidence_id'] for d in descriptors if d['category'] in registration.required_evidence_types)
    capabilities = tuple(sorted(set(registration.required_capabilities) &
                                set(context.orchestration_context.available_capabilities)))
    breakdown = {'base_relevance': config.base_relevance,
        'evidence_match': config.evidence_match_increment if family_refs else 0.,
        'category_match': config.category_match_increment if category_refs else 0.,
        'capability_match': config.capability_match_increment if capabilities else 0.,
        'evidence_match_refs': family_refs, 'category_match_refs': category_refs,
        'capability_matches': capabilities}
    raw = fsum(breakdown[k] for k in ('base_relevance', 'evidence_match', 'category_match', 'capability_match'))
    breakdown['unclipped_relevance'] = raw
    score = min(1., raw)
    breakdown['clipping_adjustment'] = score - raw
    return score, breakdown


def _select(scores, eligibility, config):
    selected, excluded = [], {}
    for identity in sorted(scores, key=lambda key: (-scores[key], key)):
        assessment = eligibility[identity]
        if assessment.status != 'eligible':
            excluded[identity] = {'reason': 'step15_ineligible', 'eligibility': bounded_plain(assessment)}
        elif scores[identity] < config.min_relevance:
            excluded[identity] = {'reason': 'below_relevance_threshold'}
        elif config.max_selected_reasoners is not None and len(selected) >= config.max_selected_reasoners:
            excluded[identity] = {'reason': 'selection_limit'}
        else:
            selected.append(identity)
    return tuple(selected), excluded


@dataclass(frozen=True)
class DynamicReasoningConnectome:
    config: ConnectomeConfig = field(default_factory=ConnectomeConfig)

    def __post_init__(self):
        if not isinstance(self.config, ConnectomeConfig):
            raise ValueError('expected ConnectomeConfig')

    def route(self, state, registry):
        if not isinstance(registry, ReasonerRegistry):
            raise ValueError('expected existing ReasonerRegistry')
        context = ConnectomeContext.from_state(state)
        orchestration = context.orchestration_context
        common = orchestration.common_evidence_state
        eligibility = {e.reasoner_id: e for e in ReasonerOrchestrator(registry).plan(orchestration)}
        nodes, scores, breakdowns, edges = {}, {}, {}, []
        for r in sorted(registry.registrations, key=lambda r: r.reasoner_id):
            nodes[r.reasoner_id] = {'node_id': r.reasoner_id, **r.metadata()}
            scores[r.reasoner_id], breakdown = _score(r, context, self.config)
            breakdowns[r.reasoner_id] = breakdown
            for rule, refs in (('evidence_match', breakdown['evidence_match_refs']),
                               ('category_match', breakdown['category_match_refs']),
                               ('capability_match', ())):
                if breakdown[rule] > 0:
                    edges.append({'edge_id': stable_id('routing-edge', orchestration.execution_id, r.reasoner_id, rule),
                        'source_context_reference': orchestration.execution_id, 'target_reasoner_id': r.reasoner_id,
                        'relation': 'context_activates_reasoner', 'rule': rule,
                        'relevance_increment': breakdown[rule], 'source_evidence_refs': refs,
                        'capability_refs': breakdown['capability_matches'] if rule == 'capability_match' else ()})
        selected, excluded = _select(scores, eligibility, self.config)
        return ReasoningGraphSnapshot(common.session_id, common.scene_id, common.timestamp,
            orchestration.execution_id, stable_id('reasoner-registry', nodes), nodes, tuple(edges),
            scores, breakdowns, selected, excluded, self.config, common.source_references,
            context.uncertainty, context.descriptors)

    def execute(self, state, registry, snapshot):
        """Explicit bridge: reject stale/tampered snapshots and recheck eligibility."""
        expected = self.route(state, registry)
        if not isinstance(snapshot, ReasoningGraphSnapshot) or snapshot.to_json() != expected.to_json():
            raise ValueError('snapshot does not match current context, registry, or routing policy')
        context = ConnectomeContext.from_state(state).orchestration_context
        registrations = {r.reasoner_id: r for r in registry.registrations}
        selected_registry = ReasonerRegistry(tuple(replace(registrations[identity], priority=order)
            for order, identity in enumerate(snapshot.selected_reasoner_ids)))
        return ReasonerOrchestrator(selected_registry).run(context)
