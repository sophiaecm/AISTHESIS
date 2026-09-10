"""Opt-in, explainable historical context; no training or calibrated probability."""
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import fsum

from ._structured import freeze, freeze_fields, identifier, number
from .evidence import EvidenceItem, EvidenceBundle, stable_id
from .prediction_records import PredictionRecord, OutcomeRecord, PredictionEvaluation


VERSION = 'experience-learning-0.1'
STRUCTURAL = ('object_class', 'motion_state', 'visibility', 'occlusion_overlap')
TERMINAL = frozenset(('supported', 'contradicted', 'partially_supported',
                      'unobservable', 'insufficient_evidence'))


def _same(a, b):
    return a is not None and type(a) is type(b) and a == b


def _bound(item, hypothesis):
    if hypothesis.track_id is not None:
        return _same(item.track_id, hypothesis.track_id)
    association = hypothesis.provenance.get('association', ())
    return (len(association) == 3 and association[0] == 'object'
            and _same(item.object_id, association[2]))


def _features(records, source_time):
    """Use issuance-time structural fields, never historical outcome features."""
    values = defaultdict(list)
    for record in records:
        value = record.get('value', {})
        stamp = value.get('timestamp', value.get('observation_timestamp', source_time))
        if stamp is None or stamp > source_time:
            continue
        for source, target in (('class_name', 'object_class'), ('motion_state', 'motion_state'),
                               ('visibility', 'visibility'), ('has_overlap_evidence', 'occlusion_overlap')):
            if value.get(source) is not None:
                values[target].append(value[source])
        modality, status = record.get('modality'), record.get('epistemic_status')
        if modality and status:
            values[f'sensory.{modality}.status'].append(status)
            if value.get('consequence') is not None:
                values[f'sensory.{modality}.consequence'].append(value['consequence'])
    result = {}
    for key, candidates in values.items():
        unique = {stable_id('feature', value): value for value in candidates}
        # Conflicting source descriptions are unavailable, not arbitrarily chosen.
        result[key] = next(iter(unique.values())) if len(unique) == 1 else None
    return result


@dataclass(frozen=True)
class ExperienceQuery:
    query_id: str
    scene_id: str
    hypothesis_id: str
    source_timestamp: float
    features: Mapping
    provenance: Mapping = field(default_factory=dict)

    def __post_init__(self):
        for name in ('query_id', 'scene_id', 'hypothesis_id'):
            identifier(getattr(self, name), name)
        number(self.source_timestamp, 'source_timestamp', nonnegative=True)
        freeze_fields(self, ('features', 'provenance'))
        if not isinstance(self.features, Mapping):
            raise ValueError('features must be a mapping')

    @classmethod
    def from_current(cls, scene, evidence, hypothesis):
        if scene.scene_id != evidence.scene_id or scene.scene_id != hypothesis.scene_id:
            raise ValueError('current source scene IDs must agree')
        if scene.timestamp != hypothesis.created_timestamp:
            raise ValueError('query requires issuance-time scene')
        records = [dict(value=x.value, modality=x.modality, epistemic_status=x.epistemic_status)
                   for x in evidence.items if x.source_type != 'experience'
                   and x.timestamp is not None and x.timestamp <= scene.timestamp
                   and _bound(x, hypothesis)]
        features = _features(records, scene.timestamp)
        features['hypothesis_type'] = hypothesis.hypothesis_type
        values = dict(scene_id=scene.scene_id, hypothesis_id=hypothesis.hypothesis_id,
            source_timestamp=scene.timestamp, features=features,
            provenance=dict(version=VERSION, source='current structured evidence only'))
        return cls(query_id=stable_id('experience-query', values), **values)


def eligibility(episode, *, current_time, query=None):
    """Fail closed on missing, inconsistent or not-yet-known record timestamps."""
    number(current_time, 'current_time', nonnegative=True)
    if episode.evaluation_status not in TERMINAL:
        return False, 'not_completed', None
    try:
        p = PredictionRecord(**episode.prediction_summary)
        o = OutcomeRecord(**episode.observation_summary)
        e = PredictionEvaluation(**episode.prediction_error)
    except (TypeError, ValueError):
        return False, 'incomplete_record_contract', None
    if (p.prediction_id != episode.prediction_id or o.prediction_id != p.prediction_id
            or e.prediction_id != p.prediction_id or e.outcome_id != o.outcome_id
            or p.hypothesis_id != episode.hypothesis_id or p.scene_id != episode.source_scene_id
            or o.scene_id != episode.target_scene_id or e.status != episode.evaluation_status
            or p.source_timestamp != episode.prediction_timestamp
            or p.source_timestamp != episode.created_timestamp
            or o.observation_timestamp != episode.observation_timestamp
            or e.evaluated_timestamp != episode.evaluated_timestamp):
        return False, 'inconsistent_record_links', None
    times = (p.source_timestamp, p.target_timestamp, o.observation_timestamp, e.evaluated_timestamp)
    if any(t is None or t >= current_time for t in times):
        return False, 'unknown_or_not_strictly_past_time', None
    if not p.source_timestamp < o.observation_timestamp <= e.evaluated_timestamp:
        return False, 'inconsistent_observation_time', None
    event_times = (p.predicted_state.get('event_timestamp'), o.observed_state.get('event_timestamp'))
    if any(t is not None and t >= current_time for t in event_times):
        return False, 'future_event_time', None
    if event_times[1] is not None and event_times[1] > e.evaluated_timestamp:
        return False, 'event_not_known_at_evaluation', None
    if query and (p.scene_id == query.scene_id or p.hypothesis_id == query.hypothesis_id):
        return False, 'current_prediction_or_scene', None
    return True, 'completed_before_query', max(times)


@dataclass(frozen=True)
class RetrievedExperience:
    episode_id: str
    query_id: str
    evaluation_status: str
    historical_features: Mapping
    matched_features: tuple
    conflicting_features: tuple
    unavailable_features: tuple
    component_matches: Mapping
    similarity_score: float
    historical_timestamps: Mapping
    evaluation_metrics: Mapping
    provenance: Mapping

    def __post_init__(self):
        number(self.similarity_score, 'similarity_score', unit=True)
        freeze_fields(self, ('historical_features', 'matched_features', 'conflicting_features',
                            'unavailable_features', 'component_matches', 'historical_timestamps',
                            'evaluation_metrics', 'provenance'))


class ExperienceRetriever:
    def __init__(self, *, top_k=3, min_similarity=.5):
        if type(top_k) is not int or not 1 <= top_k <= 8:
            raise ValueError('top_k must be an integer from 1 to 8')
        number(min_similarity, 'min_similarity', unit=True)
        self.top_k, self.min_similarity = top_k, min_similarity

    def retrieve(self, query, memory, *, current_time):
        """Memory admission and current_time must share an explicit clock domain."""
        if current_time != query.source_timestamp:
            raise ValueError('current_time must equal query issuance time')
        matches, audit = [], []
        for episode in sorted(memory.snapshot(at_time=current_time), key=lambda x: x.episode_id):
            valid, reason, _ = eligibility(episode, current_time=current_time, query=query)
            audit.append(dict(episode_id=episode.episode_id, eligible=valid, reason=reason))
            if not valid:
                continue
            p = episode.prediction_summary
            source = p['provenance'].get('hypothesis_provenance', {})
            features = _features(source.get('score_inputs', ()), p['source_timestamp'])
            features['hypothesis_type'] = p['hypothesis_type']
            matched, conflicting, unavailable, components = [], [], [], {}
            for key in sorted(set(STRUCTURAL) | set(query.features) | set(features)):
                current, historical = query.features.get(key), features.get(key)
                if current in (None, 'unknown', 'unavailable') or historical in (None, 'unknown', 'unavailable'):
                    unavailable.append(key)
                    result = 'unavailable'
                elif type(current) is type(historical) and current == historical:
                    matched.append(key)
                    result = 'matched'
                else:
                    conflicting.append(key)
                    result = 'conflicting'
                components[key] = dict(current=current, historical=historical, result=result)
            score = len(matched) / (len(matched) + len(conflicting)) if matched or conflicting else 0.
            # Matching only a generic hypothesis label or missing modalities is insufficient.
            if ('hypothesis_type' not in matched or not any(key in matched for key in STRUCTURAL)
                    or score < self.min_similarity):
                continue
            matches.append(RetrievedExperience(episode.episode_id, query.query_id,
                episode.evaluation_status.value, features, tuple(matched), tuple(conflicting), tuple(unavailable),
                components, score, dict(source_timestamp=p['source_timestamp'], target_timestamp=p['target_timestamp'],
                    observation_timestamp=episode.observation_timestamp, evaluated_timestamp=episode.evaluated_timestamp),
                episode.prediction_error['metrics'],
                dict(version=VERSION, score_policy='matched / comparable; missing excluded; heuristic, not probability',
                     historical_prediction_id=episode.prediction_id, historical_hypothesis_id=episode.hypothesis_id,
                     evidence_for=p['evidence_for'], evidence_against=p['evidence_against'],
                     source_semantics='historical prediction context; not a current observation',
                     top_k=self.top_k, min_similarity=self.min_similarity)))
        matches.sort(key=lambda x: (-x.similarity_score, x.episode_id))
        return tuple(matches[:self.top_k]), tuple(audit)


class ExperienceEvidenceProvider:
    def provide(self, scene, hypothesis, query, retrieved):
        items = []
        association = hypothesis.provenance.get('association', ())
        object_id = association[2] if len(association) == 3 and association[0] == 'object' else None
        for record in retrieved:
            if (record.query_id != query.query_id or query.scene_id != scene.scene_id
                    or query.hypothesis_id != hypothesis.hypothesis_id
                    or record.historical_features.get('hypothesis_type') != hypothesis.hypothesis_type
                    or any(t is None or t >= scene.timestamp for t in record.historical_timestamps.values())):
                raise ValueError('retrieved context does not match query')
            value = dict(source_episode_id=record.episode_id, hypothesis_id=hypothesis.hypothesis_id,
                historical_prediction_type=record.historical_features['hypothesis_type'],
                evaluation_status=record.evaluation_status, matched_features=record.matched_features,
                conflicting_features=record.conflicting_features, unavailable_features=record.unavailable_features,
                component_matches=record.component_matches, similarity_score=record.similarity_score,
                historical_evaluation_metrics=record.evaluation_metrics,
                current_observation=False, historical_context=True)
            provenance = dict(record.provenance, historical_timestamps=record.historical_timestamps,
                query_id=query.query_id, historical_features=record.historical_features,
                historical_sensory_policy='expected/inferred stays historical; never a current sensor measurement')
            items.append(EvidenceItem(stable_id('experience-evidence', scene.scene_id, value, provenance),
                scene.scene_id, 'experience', __name__ + '.ExperienceEvidenceProvider', 'historical_context',
                value, scene.timestamp, track_id=hypothesis.track_id, object_id=object_id, provenance=provenance))
        return EvidenceBundle(scene.scene_id, tuple(items), {'version': VERSION})


def experience_context(scene, evidence, hypotheses, memory=None, *, current_time, enabled=False,
                       top_k=3, max_contribution=.05):
    """Return a separate ranking view; preserve the exact original HypothesisSet.

    Absolute contribution <= .05 and relative contribution <= 5% of base score.
    No accumulation with episode count, no normalization, no selected winner.
    """
    number(max_contribution, 'max_contribution', nonnegative=True)
    if max_contribution > .05:
        raise ValueError('maximum permitted contribution is .05')
    if scene.scene_id != evidence.scene_id or scene.scene_id != hypotheses.scene_id:
        raise ValueError('current scene IDs must agree')
    if current_time != scene.timestamp or current_time != hypotheses.created_timestamp:
        raise ValueError('context must use the current issuance timestamp')
    retriever = ExperienceRetriever(top_k=top_k)
    rows, items, audits = [], [], []
    for h in sorted(hypotheses.hypotheses, key=lambda x: x.hypothesis_id):
        current = [x for x in evidence.items if x.source_type in ('physics', 'motion', 'temporal', 'occlusion', 'tracking')
                   and x.timestamp == current_time and _bound(x, h) and not x.value.get('is_predicted')]
        supporting = [x.evidence_id for x in current if h.hypothesis_type in x.supports]
        opposing = [x.evidence_id for x in current if h.hypothesis_type in x.contradicts]
        records, contribution, reason = (), 0., 'disabled'
        if enabled:
            if memory is None:
                raise ValueError('enabled experience requires ExperienceMemory')
            query = ExperienceQuery.from_current(scene, evidence, h)
            records, audit = retriever.retrieve(query, memory, current_time=current_time)
            audits.append(dict(query_id=query.query_id, hypothesis_id=h.hypothesis_id, checks=audit))
            converted = ExperienceEvidenceProvider().provide(scene, h, query, records)
            items.extend(converted.items)
            reason = ('current_contradiction' if opposing or h.evidence_against else
                      'no_current_physical_support' if not supporting else
                      'inactive_hypothesis' if h.status != 'active' else 'bounded_historical_context')
            if reason == 'bounded_historical_context':
                terms = []
                for record in records:
                    sign = {'supported': 1., 'contradicted': -1.}.get(record.evaluation_status, 0.)
                    metric = {'continued_motion': 'motion_state_match', 'object_stops': 'motion_state_match',
                              'object_becomes_occluded': 'visibility_match', 'object_reappears': 'visibility_match'
                              }.get(h.hypothesis_type, 'event_match')
                    comparable = record.evaluation_metrics.get(metric)
                    # Dissimilar structural components and unevaluable outcomes cannot influence ranking.
                    # A position-only error is not an event/motion counterexample.
                    if (not record.conflicting_features and sign
                            and comparable is (sign > 0)):
                        terms.append(sign * record.similarity_score)
                if terms:
                    bound = min(max_contribution, .05 * h.posterior_probability)
                    contribution = max(-bound, min(bound, bound * (fsum(terms) / len(terms))))
        rows.append(freeze(dict(hypothesis_id=h.hypothesis_id, original_heuristic_score=h.posterior_probability,
            experience_contribution=contribution, final_heuristic_score=h.posterior_probability + contribution,
            current_physical_support=bool(supporting), supporting_evidence_ids=tuple(sorted(supporting)),
            opposing_evidence_ids=tuple(sorted(opposing)), reason=reason,
            retrieved=tuple(freeze(record) for record in records))))
    return dict(hypotheses=hypotheses, ranking=tuple(rows),
        experience_evidence=EvidenceBundle(scene.scene_id, tuple(items)), leakage_audit=freeze(audits),
        provenance=freeze(dict(version=VERSION, enabled=enabled, max_contribution=max_contribution,
            relative_bound=.05, calibrated=False, winner_selected=False,
            base_hypotheses_modified=False, experience_is_ground_truth=False)))
