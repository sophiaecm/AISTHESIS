"""Session-only episode storage. No evaluation, inference, learning or I/O."""
from collections import OrderedDict, Counter
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
import time
from typing import Any, Mapping

from ._structured import freeze_fields, identifier, number, timestamps


class EvaluationStatus(str, Enum):
    PENDING = 'pending'
    SUPPORTED = 'supported'
    CONTRADICTED = 'contradicted'
    PARTIALLY_SUPPORTED = 'partially_supported'
    UNEVALUABLE = 'unevaluable'
    EXPIRED = 'expired'
    UNOBSERVABLE = 'unobservable'
    INSUFFICIENT_EVIDENCE = 'insufficient_evidence'


@dataclass(frozen=True)
class ExperienceEpisode:
    """Caller-supplied links and summaries, with no automatic status transitions.

    prediction_error is an optional structured metric record supplied externally;
    there is no metric or loss defined here. Missing observations imply nothing
    about evaluation_status. A Hypothesis can be supplied as hypothesis_snapshot
    and is detached into an immutable field mapping.
    """
    episode_id: str
    source_scene_id: str
    hypothesis_id: str
    created_timestamp: float
    target_scene_id: str | None = None
    trajectory_id: str | None = None
    prediction_id: str | None = None
    hypothesis_snapshot: Mapping[str, Any] = field(default_factory=dict)
    prediction_summary: Mapping[str, Any] = field(default_factory=dict)
    observation_summary: Mapping[str, Any] = field(default_factory=dict)
    prediction_timestamp: float | None = None
    observation_timestamp: float | None = None
    evaluation_status: EvaluationStatus = EvaluationStatus.PENDING
    prediction_error: Mapping[str, Any] | None = None
    evaluated_timestamp: float | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        for name in ('episode_id', 'source_scene_id', 'hypothesis_id'):
            identifier(getattr(self, name), name)
        for name in ('target_scene_id', 'trajectory_id', 'prediction_id'):
            identifier(getattr(self, name), name, optional=True)
        timestamps(self, ('created_timestamp', 'prediction_timestamp', 'observation_timestamp',
                          'evaluated_timestamp'), optional=('prediction_timestamp',
                          'observation_timestamp', 'evaluated_timestamp'))
        try:
            object.__setattr__(self, 'evaluation_status', EvaluationStatus(self.evaluation_status))
        except (ValueError, TypeError) as exc:
            raise ValueError('evaluation_status must be pending, supported, contradicted, '
                             'partially_supported, unevaluable, expired, unobservable or insufficient_evidence') from exc
        freeze_fields(self, ('hypothesis_snapshot', 'prediction_summary', 'observation_summary',
                            'prediction_error', 'provenance'))
        for name in ('hypothesis_snapshot', 'prediction_summary', 'observation_summary', 'provenance'):
            if not isinstance(getattr(self, name), Mapping):
                raise ValueError(f'{name} must be a structured mapping')
        if self.prediction_error is not None and not isinstance(self.prediction_error, Mapping):
            raise ValueError('prediction_error must be a structured mapping or None')
        for key, expected in (('hypothesis_id', self.hypothesis_id), ('scene_id', self.source_scene_id)):
            if key in self.hypothesis_snapshot and self.hypothesis_snapshot[key] != expected:
                raise ValueError(f'hypothesis_snapshot.{key} does not match episode linkage')


class ExperienceMemory:
    """At most 512 episodes, expiring 60 monotonic seconds after admission.

    Oldest insertion is evicted first; reads never refresh TTL/order. Duplicate
    episode IDs and non-null prediction IDs are rejected among currently retained
    entries, before capacity eviction. IDs can be reused after expiry/eviction.
    All operations (including cleanup on reads) hold the same RLock. recent()
    returns newest admissions first. No disk, callbacks, or inference coupling.
    """
    MAX_EPISODES = 512
    TTL_SECONDS = 60.0

    def __init__(self, *, capacity=MAX_EPISODES, clock=None):
        if type(capacity) is not int or not 1 <= capacity <= self.MAX_EPISODES:
            raise ValueError('capacity must be an integer between 1 and 512')
        self._capacity = capacity
        self._clock = time.monotonic if clock is None else clock
        self._lock = RLock()
        self._episodes = OrderedDict()
        self._prediction_ids = {}

    def _remove(self, episode_id):
        _, episode = self._episodes.pop(episode_id)
        if episode.prediction_id is not None:
            del self._prediction_ids[episode.prediction_id]

    def _cleanup(self, now):
        expired = [key for key, (added, _) in self._episodes.items()
                   if now - added >= self.TTL_SECONDS]
        for key in expired:
            self._remove(key)
        return len(expired)

    def add(self, episode: ExperienceEpisode):
        if not isinstance(episode, ExperienceEpisode):
            raise ValueError('episode must be an ExperienceEpisode')
        with self._lock:
            now = self._clock()
            self._cleanup(now)
            if episode.episode_id in self._episodes:
                raise ValueError(f'duplicate episode_id: {episode.episode_id}')
            if episode.prediction_id is not None and episode.prediction_id in self._prediction_ids:
                raise ValueError(f'duplicate prediction_id: {episode.prediction_id}')
            if len(self._episodes) == self._capacity:
                self._remove(next(iter(self._episodes)))
            self._episodes[episode.episode_id] = (now, episode)
            if episode.prediction_id is not None:
                self._prediction_ids[episode.prediction_id] = episode.episode_id

    def get(self, episode_id):
        with self._lock:
            self._cleanup(self._clock())
            record = self._episodes.get(episode_id)
            return record[1] if record else None

    def recent(self, limit=None):
        if limit is not None and (type(limit) is not int or limit < 0):
            raise ValueError('limit must be a non-negative integer or None')
        with self._lock:
            self._cleanup(self._clock())
            return tuple(item[1] for item in reversed(self._episodes.values()))[:limit]

    def snapshot(self, *, at_time):
        """Read retained, nonexpired episodes without clock calls or mutations.

        at_time must use the same clock domain as admission. For deterministic
        replay, construct this memory with an explicitly controlled replay clock.
        Unlike recent(), this read does not remove expired storage entries.
        """
        number(at_time, 'at_time', nonnegative=True)
        with self._lock:
            return tuple(episode for added, episode in self._episodes.values()
                         if 0 <= at_time - added < self.TTL_SECONDS)

    def cleanup(self):
        with self._lock:
            return self._cleanup(self._clock())

    def summary(self):
        with self._lock:
            self._cleanup(self._clock())
            return dict(count=len(self._episodes), capacity=self._capacity,
                        ttl_seconds=self.TTL_SECONDS,
                        by_status=dict(Counter(e.evaluation_status.value
                                               for _, e in self._episodes.values())))
