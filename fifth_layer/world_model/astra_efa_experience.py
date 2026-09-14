"""Deterministic longitudinal representation, with no inference or learning feedback."""
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
import json

from ._structured import freeze, identifier, number
from .common_evidence_state import bounded_plain
from .evidence import stable_id
from .experience_memory import ExperienceEpisode
from .prediction_records import PredictionRecord, OutcomeRecord, PredictionEvaluation

VERSION = 'astra-efa-experience-v0.1'
TERMINAL = ('supported', 'partially_supported', 'contradicted',
            'unobservable', 'insufficient_evidence')
COMPARABLE = TERMINAL[:3]
POLICY = {'representation_only': True, 'learning': False, 'inference_feedback': False,
          'world_transformation': False, 'ground_truth': False,
          'metric_aggregation': 'deferred', 'edis': 'deferred'}


class _Record:
    def to_dict(self):
        return bounded_plain(self)

    def to_json(self):
        return json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'), allow_nan=False)


def _set(record, **values):
    for name, value in values.items():
        object.__setattr__(record, name, value)


def _identity(record, name):
    data = record.to_dict()
    data.pop(name, None)
    _set(record, **{name: stable_id(name, VERSION, data)})


def _audit(value, session, cutoff, *, observed=False):
    """Check declared source times recursively; targets are prospective metadata."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == 'session_id' and child != session:
                raise ValueError('session mismatch in source metadata')
            if (key == 'timestamp' or key.endswith('_timestamp')) and key != 'target_timestamp' and child is not None:
                number(child, key, nonnegative=True)
                if child > cutoff:
                    raise ValueError('future source timestamp')
            if observed and ((key == 'is_predicted' and child) or
                    (key == 'observed' and child is not True) or
                    (key == 'epistemic_status' and child != 'observed')):
                raise ValueError('non-observed source cannot become observation')
            _audit(child, session, cutoff, observed=observed)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _audit(child, session, cutoff, observed=observed)


@dataclass(frozen=True)
class AstraEncounter(_Record):
    """Construct from a complete detached episode; all projections are derived."""
    source_episode: Mapping
    encounter_id: str = field(default='', init=False)
    session_id: str = field(init=False)
    source_episode_id: str = field(init=False)
    prediction_id: str = field(init=False)
    hypothesis_id: str = field(init=False)
    source_scene_id: str = field(init=False)
    target_scene_id: str | None = field(init=False)
    prediction_timestamp: float = field(init=False)
    observation_timestamp: float = field(init=False)
    evaluated_timestamp: float = field(init=False)
    evaluation_status: str = field(init=False)
    prediction_summary: Mapping = field(init=False)
    observation_summary: Mapping = field(init=False)
    evaluation_summary: Mapping = field(init=False)
    source_belief_state_id: str | None = field(init=False)
    source_future_id: str | None = field(init=False)
    provenance: Mapping = field(init=False)
    schema_version: str = field(default=VERSION, init=False)

    def __post_init__(self):
        try:
            raw = bounded_plain(self.source_episode)
            if set(raw) != {f.name for f in fields(ExperienceEpisode)}:
                raise ValueError('complete episode summary required')
            ep = ExperienceEpisode(**raw)
            records = []
            for cls, summary in ((PredictionRecord, ep.prediction_summary),
                                 (OutcomeRecord, ep.observation_summary),
                                 (PredictionEvaluation, ep.prediction_error)):
                if not isinstance(summary, Mapping) or set(summary) != {f.name for f in fields(cls)}:
                    raise ValueError('complete record summary required')
                records.append(cls(**summary))
            p, o, e = records
        except (TypeError, KeyError) as exc:
            raise ValueError('incomplete episode record') from exc
        if e.status.value not in TERMINAL or ep.evaluation_status != e.status:
            raise ValueError('completed consistent status required')
        if (ep.prediction_id != p.prediction_id or o.prediction_id != p.prediction_id or
                e.prediction_id != p.prediction_id or e.outcome_id != o.outcome_id or
                ep.hypothesis_id != p.hypothesis_id or ep.source_scene_id != p.scene_id or
                ep.target_scene_id != o.scene_id or ep.trajectory_id != p.trajectory_id):
            raise ValueError('inconsistent episode identity')
        if (ep.created_timestamp != p.source_timestamp or ep.prediction_timestamp != p.source_timestamp or
                ep.observation_timestamp != o.observation_timestamp or ep.evaluated_timestamp != e.evaluated_timestamp):
            raise ValueError('inconsistent episode timestamps')
        for stamp in (p.source_timestamp, o.observation_timestamp, e.evaluated_timestamp):
            number(stamp, 'required timestamp', nonnegative=True)
        if not (p.source_timestamp < o.observation_timestamp <= e.evaluated_timestamp) or p.horizon_seconds <= 0:
            raise ValueError('prediction must precede observation, then evaluation')
        session = p.provenance.get('session_id')
        identifier(session, 'prediction provenance session_id')
        _audit(p.provenance, session, p.source_timestamp)
        _audit(ep.hypothesis_snapshot, session, p.source_timestamp)
        _audit(o.observed_state, session, o.observation_timestamp, observed=True)
        _audit(o.provenance, session, o.observation_timestamp)
        _audit(o.provenance.get('sources', ()), session, o.observation_timestamp, observed=True)
        _audit(e.metrics, session, e.evaluated_timestamp)
        _audit(e.provenance, session, e.evaluated_timestamp)
        _audit(ep.provenance, session, e.evaluated_timestamp)
        if p.provenance.get('hypothesis_id', p.hypothesis_id) != p.hypothesis_id:
            raise ValueError('prediction provenance hypothesis mismatch')
        for name in ('belief_state_id', 'source_future_id'):
            identifier(p.provenance.get(name), name, optional=True)
        if e.status.value in COMPARABLE:
            identity = 'track_id' if p.track_id is not None else 'object_id'
            a, b = getattr(p, identity), getattr(o, identity)
            if o.association_status != 'matched' or a is None or type(a) is not type(b) or a != b:
                raise ValueError('comparable evaluation requires matching association')
        _set(self, source_episode=freeze(raw), session_id=session,
             source_episode_id=ep.episode_id, prediction_id=p.prediction_id,
             hypothesis_id=p.hypothesis_id, source_scene_id=p.scene_id, target_scene_id=o.scene_id,
             prediction_timestamp=p.source_timestamp, observation_timestamp=o.observation_timestamp,
             evaluated_timestamp=e.evaluated_timestamp, evaluation_status=e.status.value,
             prediction_summary=freeze(p), observation_summary=freeze(o), evaluation_summary=freeze(e),
             source_belief_state_id=p.provenance.get('belief_state_id'),
             source_future_id=p.provenance.get('source_future_id'),
             provenance=freeze(dict(POLICY, source_episode_id=ep.episode_id,
                                    evaluation_id=e.evaluation_id, outcome_id=o.outcome_id)))
        _identity(self, 'encounter_id')


class EncounterEncoder:
    @staticmethod
    def encode(episode):
        if not isinstance(episode, ExperienceEpisode):
            raise ValueError('ExperienceEpisode required')
        return AstraEncounter(freeze(episode))


@dataclass(frozen=True)
class AstraSelfState(_Record):
    session_id: str
    timestamp: float
    completed_experience_ids: tuple = ()
    completed_episode_ids: tuple = ()
    completed_prediction_ids: tuple = ()
    evaluation_statuses: tuple = ()
    previous_state_id: str | None = None
    state_id: str = field(default='', init=False)
    sequence_index: int = field(init=False)
    evaluation_counts: Mapping = field(init=False)
    cumulative_comparable_predictions: int = field(init=False)
    cumulative_supported: int = field(init=False)
    cumulative_partially_supported: int = field(init=False)
    cumulative_contradicted: int = field(init=False)
    cumulative_unobservable: int = field(init=False)
    cumulative_insufficient_evidence: int = field(init=False)
    cumulative_error_summary: Mapping = field(init=False)
    last_experience_id: str | None = field(init=False)
    provenance: Mapping = field(init=False)
    schema_version: str = field(default=VERSION, init=False)

    def __post_init__(self):
        identifier(self.session_id, 'session_id')
        number(self.timestamp, 'timestamp', nonnegative=True)
        for name in ('completed_experience_ids', 'completed_episode_ids', 'completed_prediction_ids', 'evaluation_statuses'):
            value = getattr(self, name)
            if not isinstance(value, (tuple, list)):
                raise ValueError('ordered state history required')
            for item in value:
                identifier(item, name)
            if name != 'evaluation_statuses' and len(set(value)) != len(value):
                raise ValueError('duplicate state history ID')
            _set(self, **{name: tuple(value)})
        n = len(self.completed_experience_ids)
        if any(len(getattr(self, name)) != n for name in
               ('completed_episode_ids', 'completed_prediction_ids', 'evaluation_statuses')):
            raise ValueError('state histories must align')
        if any(s not in TERMINAL for s in self.evaluation_statuses):
            raise ValueError('invalid state status')
        identifier(self.previous_state_id, 'previous_state_id', optional=n == 0)
        if n == 0 and self.previous_state_id is not None:
            raise ValueError('initial state cannot have a predecessor')
        counts = {s: self.evaluation_statuses.count(s) for s in sorted(TERMINAL)}
        _set(self, sequence_index=n, evaluation_counts=freeze(counts),
             cumulative_comparable_predictions=sum(counts[s] for s in COMPARABLE),
             cumulative_error_summary=freeze({}),
             last_experience_id=self.completed_experience_ids[-1] if n else None,
             provenance=freeze(POLICY), **{'cumulative_' + s: counts[s] for s in TERMINAL})
        _identity(self, 'state_id')


def _experience_id(previous, encounter):
    # Pre-state + encounter avoids a cyclic dependency on the post-state ID.
    return stable_id('experience_id', VERSION, previous.state_id, encounter.encounter_id)


class RecursiveUpdater:
    @staticmethod
    def update(previous_state, encounter):
        if not isinstance(previous_state, AstraSelfState) or not isinstance(encounter, AstraEncounter):
            raise ValueError('SelfState and Encounter required')
        p, e = previous_state, encounter
        if p.session_id != e.session_id:
            raise ValueError('session mismatch')
        if e.source_episode_id in p.completed_episode_ids or e.prediction_id in p.completed_prediction_ids:
            raise ValueError('duplicate episode or prediction')
        if e.evaluated_timestamp <= p.timestamp:
            raise ValueError('evaluation timestamp must strictly increase')
        return AstraSelfState(p.session_id, e.evaluated_timestamp,
            p.completed_experience_ids + (_experience_id(p, e),),
            p.completed_episode_ids + (e.source_episode_id,),
            p.completed_prediction_ids + (e.prediction_id,),
            p.evaluation_statuses + (e.evaluation_status,), p.state_id)


@dataclass(frozen=True)
class AstraTransformation(_Record):
    pre_state_id: str
    post_state_id: str
    encounter_id: str
    sequence_index: int
    changed_fields: tuple
    unchanged_fields: tuple
    delta_summary: Mapping
    provenance: Mapping = field(default_factory=lambda: freeze(POLICY), init=False)
    transformation_id: str = field(default='', init=False)
    schema_version: str = field(default=VERSION, init=False)

    def __post_init__(self):
        for name in ('pre_state_id', 'post_state_id', 'encounter_id'):
            identifier(getattr(self, name), name)
        if type(self.sequence_index) is not int or self.sequence_index < 1:
            raise ValueError('positive sequence required')
        for name in ('changed_fields', 'unchanged_fields'):
            values = getattr(self, name)
            if not isinstance(values, (tuple, list)) or tuple(values) != tuple(sorted(set(values))):
                raise ValueError('sorted unique field names required')
            _set(self, **{name: tuple(values)})
        if set(self.changed_fields) & set(self.unchanged_fields) or set(self.delta_summary) != set(self.changed_fields):
            raise ValueError('inconsistent difference inventory')
        if set(self.changed_fields) | set(self.unchanged_fields) != {f.name for f in fields(AstraSelfState)}:
            raise ValueError('complete state field inventory required')
        for delta in self.delta_summary.values():
            if not isinstance(delta, Mapping) or set(delta) != {'before', 'after'} or delta['before'] == delta['after']:
                raise ValueError('changed fields require different before/after values')
        _set(self, delta_summary=freeze(bounded_plain(self.delta_summary)))
        _identity(self, 'transformation_id')


class TransformationEncoder:
    @staticmethod
    def encode(pre_state, encounter, post_state):
        if RecursiveUpdater.update(pre_state, encounter) != post_state:
            raise ValueError('post-state must be the mechanical recursive update')
        before, after = pre_state.to_dict(), post_state.to_dict()
        changed = tuple(k for k in sorted(before) if before[k] != after[k])
        unchanged = tuple(k for k in sorted(before) if before[k] == after[k])
        return AstraTransformation(pre_state.state_id, post_state.state_id, encounter.encounter_id,
            post_state.sequence_index, changed, unchanged,
            {k: {'before': before[k], 'after': after[k]} for k in changed})


@dataclass(frozen=True)
class AstraExperienceState(_Record):
    encounter: AstraEncounter
    pre_state: AstraSelfState
    post_state: AstraSelfState = field(init=False)
    transformation: AstraTransformation = field(init=False)
    experience_id: str = field(init=False)
    source_episode_id: str = field(init=False)
    sequence_index: int = field(init=False)
    provenance: Mapping = field(default_factory=lambda: freeze(POLICY), init=False)
    schema_version: str = field(default=VERSION, init=False)

    def __post_init__(self):
        post = RecursiveUpdater.update(self.pre_state, self.encounter)
        _set(self, post_state=post,
             transformation=TransformationEncoder.encode(self.pre_state, self.encounter, post),
             experience_id=_experience_id(self.pre_state, self.encounter),
             source_episode_id=self.encounter.source_episode_id, sequence_index=post.sequence_index)


@dataclass(frozen=True)
class AstraExperienceTrajectory(_Record):
    initial_state: AstraSelfState
    experiences: tuple = ()
    trajectory_id: str = field(default='', init=False)
    session_id: str = field(init=False)
    current_state: AstraSelfState = field(init=False)
    raw_episode_ids: tuple = field(init=False)
    experience_ids: tuple = field(init=False)
    provenance: Mapping = field(default_factory=lambda: freeze(POLICY), init=False)
    schema_version: str = field(default=VERSION, init=False)

    def __post_init__(self):
        if not isinstance(self.initial_state, AstraSelfState) or self.initial_state.sequence_index != 0:
            raise ValueError('neutral initial state required')
        if not isinstance(self.experiences, (tuple, list)):
            raise ValueError('ordered experiences required')
        previous = self.initial_state
        for experience in self.experiences:
            if not isinstance(experience, AstraExperienceState) or experience.pre_state != previous:
                raise ValueError('broken trajectory lineage')
            previous = experience.post_state
        _set(self, experiences=tuple(self.experiences), session_id=previous.session_id,
             current_state=previous, raw_episode_ids=previous.completed_episode_ids,
             experience_ids=previous.completed_experience_ids)
        _identity(self, 'trajectory_id')

    @property
    def raw_history(self):
        return self.raw_episode_ids

    @property
    def experience_history(self):
        return tuple((e.experience_id, e.transformation.transformation_id) for e in self.experiences)


class AstraEFAExperienceBuilder:
    @staticmethod
    def initialize(*, session_id, timestamp):
        """Neutral software state at an explicit caller-supplied timestamp."""
        return AstraSelfState(session_id, timestamp)

    encode = staticmethod(EncounterEncoder.encode)
    update = staticmethod(RecursiveUpdater.update)

    @staticmethod
    def append(trajectory, episode):
        if not isinstance(trajectory, AstraExperienceTrajectory):
            raise ValueError('trajectory required')
        experience = AstraExperienceState(EncounterEncoder.encode(episode), trajectory.current_state)
        return AstraExperienceTrajectory(trajectory.initial_state, trajectory.experiences + (experience,))

    @classmethod
    def build(cls, episodes, *, session_id, timestamp):
        trajectory = AstraExperienceTrajectory(cls.initialize(session_id=session_id, timestamp=timestamp))
        for episode in episodes:
            trajectory = cls.append(trajectory, episode)
        return trajectory

    replay = build
