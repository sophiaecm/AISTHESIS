"""Opt-in trajectory context. Scores are heuristic views, never Bayesian updates."""
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from math import fsum
import json

from ._structured import freeze, identifier, number
from .common_evidence_state import bounded_plain
from .evidence import stable_id
from .bayesian_belief_state import BayesianBeliefState
from .hybrid_world_state import _alignment
from .astra_efa_experience import (
    AstraEncounter, AstraExperienceState, AstraExperienceTrajectory,
    AstraEFAExperienceBuilder, TERMINAL,
)

VERSION = 'experience-learning-v0.2'
FEATURES = ('assumption_types', 'branch_family', 'has_consequence_references',
            'horizon_kind', 'object_count', 'predicted_fields', 'predicted_values')
DIRECTION = {'supported': 1, 'contradicted': -1, 'partially_supported': 0,
             'unobservable': 0, 'insufficient_evidence': 0}
POLICY = dict(calibrated=False, probability_update=False, Bayesian_feedback=False,
    winner_selected=False, history_is_ground_truth=False, current_observation=False,
    experience_learning_mode='bounded_historical_context', truth_decision='not_performed',
    similarity_is_probability=False, score_is_probability=False, recency_decay=False)
_VALUES = {'motion_state': ('moving',), 'contact_status': ('possible_contact', 'no_contact_assumed'),
           'continuity_status': ('possible_continuity',)}
_ASSUMPTIONS = ('motion_persists', 'contact_within_horizon', 'no_contact_within_horizon',
                'continuity_resumes', 'outcome_unresolved')


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
            elif isinstance(value, list):
                object.__setattr__(self, f.name, tuple(value))
        data = self.to_dict()
        id_field = next(f.name for f in fields(self) if f.name in ('query_id', 'retrieval_id', 'view_id') and not f.init)
        data.pop(id_field)
        object.__setattr__(self, id_field, stable_id(id_field, VERSION, data))


def _ordered_strings(value):
    if not isinstance(value, (tuple, list)) or any(type(v) is not str or not v for v in value):
        return None
    return tuple(sorted(set(value)))


def _features(family, assumptions, changes, horizon, consequences, object_count=None):
    """Whitelisted semantic values only; never compare reference values or probabilities."""
    types = None
    if isinstance(assumptions, (tuple, list)) and assumptions and all(
            isinstance(a, Mapping) and a.get('assumption_type') in _ASSUMPTIONS for a in assumptions):
        types = tuple(sorted({a['assumption_type'] for a in assumptions}))
    names, values = None, None
    if isinstance(changes, (tuple, list)) and changes and all(isinstance(c, Mapping) for c in changes):
        known = [c for c in changes if type(c.get('field')) is str and
                 (c['field'] in _VALUES or c['field'] == 'consequence_reference')]
        names = tuple(sorted({c['field'] for c in known})) or None
        semantic = [c for c in known if c['field'] in _VALUES]
        if semantic and all(c.get('value') in _VALUES[c['field']] and
                            c.get('epistemic_status') == 'branch_assumption' for c in semantic):
            values = tuple(sorted((c['field'], c['value'], c['epistemic_status']) for c in semantic))
    refs = _ordered_strings(consequences)
    return freeze(dict(branch_family=family if family in ('motion_continuation', 'contact_resolution',
        'discontinuity_resolution') else None, assumption_types=types, predicted_fields=names,
        predicted_values=values, horizon_kind=horizon if horizon in ('next_transition', 'short_horizon') else None,
        object_count=object_count, has_consequence_references=None if refs is None else bool(refs)))


def historical_features(encounter):
    p = encounter.prediction_summary['provenance']
    return _features(p.get('source_branch_family'), p.get('source_assumptions'),
        p.get('source_predicted_changes'), p.get('source_horizon_kind'), p.get('source_consequence_ids'))


def _belief(value):
    if not isinstance(value, BayesianBeliefState) or value.schema_version != 'bayesian-belief-state-v0.1':
        raise ValueError('BayesianBeliefState v0.1 required')
    # Re-run the frozen contracts, including current declared time/session checks.
    checked = replace(value, beliefs=tuple(replace(b, source_future=replace(b.source_future)) for b in value.beliefs))
    if checked != value:
        raise ValueError('inconsistent current belief')
    _alignment(bounded_plain(value), value)


@dataclass(frozen=True)
class TrajectoryExperienceQuery(_Record):
    scene_id: str
    session_id: str
    timestamp: float | None
    belief_state_id: str
    hypothesis_id: str
    source_future_id: str
    branch_family: str
    assumption_types: tuple
    predicted_fields: tuple
    source_constraint_ids: tuple
    source_consequence_ids: tuple
    current_structural_features: Mapping
    provenance: Mapping
    query_id: str = field(default='', init=False)
    schema_version: str = field(default=VERSION, init=False)

    def __post_init__(self):
        for name in ('scene_id', 'session_id', 'belief_state_id', 'hypothesis_id', 'source_future_id'):
            identifier(getattr(self, name), name)
        if self.timestamp is not None:
            number(self.timestamp, 'query timestamp', nonnegative=True)
        if set(self.current_structural_features) != set(FEATURES):
            raise ValueError('exact structural feature inventory required')
        super().__post_init__()

    @classmethod
    def from_current(cls, belief_state, hypothesis_id):
        _belief(belief_state)
        b = next((b for b in belief_state.beliefs if b.hypothesis_id == hypothesis_id), None)
        if b is None:
            raise ValueError('unknown current hypothesis')
        f = b.source_future
        features = _features(f.branch_family, bounded_plain(f.assumptions), f.predicted_changes,
            f.horizon_kind.value, f.source_consequence_ids, len(f.object_ids))
        return cls(b.scene_id, b.session_id, b.timestamp, belief_state.belief_state_id,
            b.hypothesis_id, f.future_id, f.branch_family, features['assumption_types'],
            features['predicted_fields'], f.source_constraint_ids, f.source_consequence_ids, features,
            dict(POLICY, source_future_schema=f.schema_version, source_belief_schema=b.schema_version,
                predicted_changes=bounded_plain(f.predicted_changes), object_ids=f.object_ids,
                source_future_provenance=f.provenance, source_belief_provenance=b.provenance,
                prior_probability=b.prior_probability, posterior_probability=b.posterior_probability,
                source_evidence_ids=b.source_evidence_ids, coordinate_frame_id=b.coordinate_frame_id))


def structural_comparison(current, historical):
    components, matched, conflicting, unavailable = {}, [], [], []
    for name in FEATURES:
        a, b = current.get(name), historical.get(name)
        if a is None or b is None:
            result = 'unavailable'
            unavailable.append(name)
        elif freeze(a) == freeze(b) and type(a) is type(b):
            result = 'matched'
            matched.append(name)
        else:
            result = 'conflicting'
            conflicting.append(name)
        components[name] = dict(current=a, historical=b, result=result)
    score = len(matched) / (len(matched) + len(conflicting)) if matched or conflicting else 0.
    meaningful = ('predicted_values' in matched or ('assumption_types' in matched and
                   current.get('assumption_types') != ('outcome_unresolved',)))
    reason = ('branch_family_mismatch_or_unavailable' if 'branch_family' not in matched else
              'no_meaningful_structural_match' if not meaningful else 'structurally_relevant')
    return score, tuple(matched), tuple(conflicting), tuple(unavailable), freeze(components), reason


def _trajectory(value):
    if not isinstance(value, AstraExperienceTrajectory) or value.schema_version != 'astra-efa-experience-v0.1':
        raise ValueError('AstraExperienceTrajectory v0.1 required when enabled')
    previous = AstraEFAExperienceBuilder.initialize(session_id=value.session_id, timestamp=value.initial_state.timestamp)
    if previous != value.initial_state:
        raise ValueError('invalid trajectory initialization')
    rebuilt = []
    for experience in value.experiences:
        if not isinstance(experience, AstraExperienceState):
            raise ValueError('invalid trajectory experience')
        expected = AstraExperienceState(AstraEncounter(experience.encounter.source_episode), previous)
        if expected != experience:
            raise ValueError('broken trajectory lineage or transformation')
        rebuilt.append(expected)
        previous = expected.post_state
    if AstraExperienceTrajectory(value.initial_state, tuple(rebuilt)) != value:
        raise ValueError('inconsistent trajectory identity or history')


@dataclass(frozen=True)
class RetrievedTrajectoryExperience(_Record):
    query_id: str
    experience_id: str
    source_episode_id: str
    prediction_id: str
    historical_status: str
    similarity_score: float
    matched_features: tuple
    conflicting_features: tuple
    unavailable_features: tuple
    structural_comparison: Mapping
    historical_timestamps: Mapping
    historical_prediction_metrics: Mapping
    historical_transformation_id: str
    pre_state_id: str
    post_state_id: str
    sequence_index: int
    provenance: Mapping
    retrieval_id: str = field(default='', init=False)
    schema_version: str = field(default=VERSION, init=False)

    def __post_init__(self):
        number(self.similarity_score, 'similarity', unit=True)
        if self.historical_status not in TERMINAL:
            raise ValueError('terminal historical status required')
        super().__post_init__()


class TrajectoryExperienceRetriever:
    def __init__(self, *, top_k=3, min_similarity=.5):
        if type(top_k) is not int or not 1 <= top_k <= 3:
            raise ValueError('top_k must be an integer from 1 to 3')
        number(min_similarity, 'min_similarity', unit=True)
        self.top_k, self.min_similarity = top_k, min_similarity

    def retrieve(self, query, trajectory):
        if not isinstance(query, TrajectoryExperienceQuery):
            raise ValueError('TrajectoryExperienceQuery required')
        _trajectory(trajectory)
        return self._retrieve_validated(query, trajectory)

    def _retrieve_validated(self, query, trajectory):
        candidates, audits = [], []
        seen = {name: set() for name in ('experience_id', 'source_episode_id', 'prediction_id', 'outcome_id')}
        for experience in trajectory.experiences:
            e = experience.encounter
            keys = dict(experience_id=experience.experience_id, source_episode_id=e.source_episode_id,
                        prediction_id=e.prediction_id, outcome_id=e.observation_summary['outcome_id'])
            duplicate = any(value in seen[name] for name, value in keys.items())
            for name, value in keys.items():
                seen[name].add(value)
            p = e.prediction_summary
            times = dict(prediction_timestamp=e.prediction_timestamp, target_timestamp=p['target_timestamp'],
                observation_timestamp=e.observation_timestamp, evaluated_timestamp=e.evaluated_timestamp)
            # Additional prospective event times must also be past, as in v0.1.
            event_time = p['predicted_state'].get('event_timestamp')
            reason = ('session_mismatch' if e.session_id != query.session_id else
                      'unknown_query_timestamp' if query.timestamp is None else
                      'not_strictly_past' if any(t >= query.timestamp for t in times.values()) or
                          (event_time is not None and event_time >= query.timestamp) else
                      'duplicate_history' if duplicate else 'completed_before_query')
            eligible = reason == 'completed_before_query'
            audit = dict(keys, query_id=query.query_id, eligible=eligible, reason=reason,
                         retrieved=False, historical_timestamps=times)
            if eligible:
                score, matched, conflict, unavailable, components, reason = structural_comparison(
                    query.current_structural_features, historical_features(e))
                if reason == 'structurally_relevant' and score < self.min_similarity:
                    reason = 'below_min_similarity'
                audit.update(reason=reason, similarity_score=score, structural_comparison=components)
                if reason == 'structurally_relevant':
                    candidates.append(RetrievedTrajectoryExperience(query.query_id, experience.experience_id,
                        e.source_episode_id, e.prediction_id, e.evaluation_status, score, matched, conflict,
                        unavailable, components, times, e.evaluation_summary['metrics'],
                        experience.transformation.transformation_id, experience.pre_state.state_id,
                        experience.post_state.state_id, experience.sequence_index,
                        dict(POLICY, source_type='historical_experience_context',
                            source_encounter_id=e.encounter_id, source_evaluation_id=e.evaluation_summary['evaluation_id'],
                            source_outcome_id=keys['outcome_id'], historical_hypothesis_id=e.hypothesis_id,
                            historical_prediction_provenance=p['provenance'],
                            historical_evaluation_provenance=e.evaluation_summary['provenance'],
                            trajectory_schema=trajectory.schema_version, top_k=self.top_k,
                            min_similarity=self.min_similarity, direction=DIRECTION[e.evaluation_status])))
            audits.append(audit)
        candidates.sort(key=lambda r: (-r.similarity_score, -r.historical_timestamps['evaluated_timestamp'], r.experience_id))
        selected = tuple(candidates[:self.top_k])
        ids = {r.experience_id for r in selected}
        for audit in audits:
            if audit['reason'] == 'structurally_relevant':
                audit.update(retrieved=audit['experience_id'] in ids,
                             reason='retrieved' if audit['experience_id'] in ids else 'outside_top_k')
        return selected, freeze(audits)


def _current_gate(state, belief):
    if state.timestamp is None:
        return 'unknown_query_timestamp'
    if state.update_status.value in ('unavailable', 'indeterminate', 'invalid_evidence'):
        return 'current_belief_' + state.update_status.value
    future = belief.source_future
    if 'source_constraint_conflict_or_ambiguity' in future.uncertainty:
        return 'current_source_conflict'
    if future.status.value in ('unsupported', 'unavailable', 'indeterminate'):
        return 'current_future_' + future.status.value
    if belief.posterior_probability is None:
        return 'posterior_unavailable'
    # Frozen Step 19 has no signed support/oppose field. Never reinterpret its
    # likelihoods as opposing evidence. Explicit branch conflict markers do exist.
    return None


@dataclass(frozen=True)
class ExperienceLearningRow(_Record):
    hypothesis_id: str
    source_future_id: str
    base_posterior_probability: float | None
    historical_contribution: float
    experience_informed_score: float | None
    influence_status: str
    retrieved_experiences: tuple
    provenance: Mapping
    schema_version: str = field(default=VERSION, init=False)

    def __post_init__(self):
        number(self.historical_contribution, 'historical contribution')
        base = self.base_posterior_probability
        if base is None:
            if self.historical_contribution != 0 or self.experience_informed_score is not None:
                raise ValueError('missing posterior cannot fabricate score')
        else:
            number(base, 'base posterior', unit=True)
            if abs(self.historical_contribution) > min(.05, .05 * base):
                raise ValueError('historical contribution exceeds bound')
            if self.experience_informed_score != base + self.historical_contribution:
                raise ValueError('score must equal base plus bounded contribution')
        object.__setattr__(self, 'retrieved_experiences', tuple(self.retrieved_experiences))
        object.__setattr__(self, 'provenance', freeze(bounded_plain(self.provenance)))


@dataclass(frozen=True)
class ExperienceLearningViewV02(_Record):
    scene_id: str
    session_id: str
    timestamp: float | None
    source_belief_state_id: str
    source_trajectory_id: str | None
    enabled: bool
    queries: tuple
    rows: tuple
    retrieval_audit: tuple
    provenance: Mapping
    view_id: str = field(default='', init=False)
    schema_version: str = field(default=VERSION, init=False)

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError('enabled must be boolean')
        ids = tuple(r.hypothesis_id for r in self.rows)
        if ids != tuple(sorted(set(ids))):
            raise ValueError('rows must be unique and sorted by hypothesis_id')
        object.__setattr__(self, 'retrieval_audit', freeze(bounded_plain(self.retrieval_audit)))
        super().__post_init__()


def experience_learning_v02(belief_state, trajectory=None, *, enabled=False,
                            top_k=3, min_similarity=.5, max_absolute_contribution=.05):
    """Separate bounded view; disabled mode never reads or retrieves history."""
    _belief(belief_state)
    if type(enabled) is not bool:
        raise ValueError('enabled must be boolean')
    retriever = TrajectoryExperienceRetriever(top_k=top_k, min_similarity=min_similarity)
    number(max_absolute_contribution, 'max_absolute_contribution', nonnegative=True)
    if max_absolute_contribution > .05:
        raise ValueError('maximum absolute contribution is .05')
    if enabled:
        _trajectory(trajectory)
    queries, rows, audits = [], [], []
    for belief in sorted(belief_state.beliefs, key=lambda b: b.hypothesis_id):
        query = TrajectoryExperienceQuery.from_current(belief_state, belief.hypothesis_id)
        queries.append(query)
        retrieved, audit = (), ()
        if enabled:
            retrieved, audit = retriever._retrieve_validated(query, trajectory)
            audits.append(dict(query_id=query.query_id, hypothesis_id=belief.hypothesis_id, checks=audit))
        base = belief.posterior_probability
        bound = 0. if base is None else min(max_absolute_contribution, .05 * base)
        gate = 'disabled' if not enabled else _current_gate(belief_state, belief)
        # Retain neutral terms in the denominator; they cannot supply direction.
        terms = tuple(DIRECTION[r.historical_status] * r.similarity_score for r in retrieved)
        mean = fsum(terms) / len(terms) if terms else 0.
        contribution = 0. if gate else max(-bound, min(bound, bound * mean))
        status = gate or ('no_relevant_history' if not retrieved else
                          'neutral_historical_context' if mean == 0 else 'bounded_historical_context')
        rows.append(ExperienceLearningRow(belief.hypothesis_id, belief.source_future_id, base, contribution,
            None if base is None else base + contribution, status, retrieved,
            dict(POLICY, enabled=enabled, query_id=query.query_id, bound=bound,
                max_absolute_contribution=max_absolute_contribution, relative_bound=.05,
                directional_terms=terms, mean_directional_term=mean, suppression_reason=gate,
                formula='clamp(min(max_absolute, .05 * base) * mean(direction * similarity), -bound, bound)',
                current_future_status=belief.source_future.status.value,
                current_belief_status=belief.update_status.value)))
    return ExperienceLearningViewV02(belief_state.scene_id, belief_state.session_id, belief_state.timestamp,
        belief_state.belief_state_id, trajectory.trajectory_id if enabled else None, enabled,
        tuple(queries), tuple(rows), tuple(audits), dict(POLICY, enabled=enabled,
            retrieval_performed=enabled, top_k=top_k, min_similarity=min_similarity,
            max_absolute_contribution=max_absolute_contribution, relative_bound=.05,
            same_session_only=True, normalized_distribution=False))
