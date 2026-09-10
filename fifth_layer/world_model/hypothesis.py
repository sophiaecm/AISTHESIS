"""Hypothesis contracts only; no hypothesis generation or inference."""
from dataclasses import dataclass, field
from enum import Enum
from math import fsum
from types import MappingProxyType
from typing import Any, Mapping

from ._structured import freeze_fields, identifier, number, timestamps


class HypothesisStatus(str, Enum):
    ACTIVE = 'active'
    SUPPORTED = 'supported'
    CONTRADICTED = 'contradicted'
    UNEVALUABLE = 'unevaluable'
    EXPIRED = 'expired'


@dataclass(frozen=True)
class Hypothesis:
    hypothesis_id: str
    scene_id: str
    hypothesis_type: str
    statement: str
    prior_probability: float
    posterior_probability: float
    confidence: float
    created_timestamp: float
    horizon_seconds: float
    target_timestamp: float
    track_id: int | str | None = None
    evidence_for: tuple = ()
    evidence_against: tuple = ()
    assumptions: tuple = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    status: HypothesisStatus = HypothesisStatus.ACTIVE

    def __post_init__(self):
        for name in ('hypothesis_id', 'scene_id', 'hypothesis_type', 'statement'):
            identifier(getattr(self, name), name)
        if self.track_id is not None and type(self.track_id) not in (int, str):
            raise ValueError('track_id must be an integer, string or None')
        for name in ('prior_probability', 'posterior_probability', 'confidence'):
            number(getattr(self, name), name, unit=True)
        timestamps(self, ('created_timestamp', 'horizon_seconds', 'target_timestamp'))
        try:
            object.__setattr__(self, 'status', HypothesisStatus(self.status))
        except (ValueError, TypeError) as exc:
            raise ValueError('status must be active, supported, contradicted, unevaluable or expired') from exc
        freeze_fields(self, ('evidence_for', 'evidence_against', 'assumptions', 'provenance'))


@dataclass(frozen=True)
class HypothesisSet:
    """Raw hypotheses plus an explicit read-only normalization view.

    active_sum normalizes ACTIVE posteriors only. Non-active hypotheses remain
    in hypotheses but are absent from normalized_posteriors. No copies overwrite
    prior or posterior values. Equal probabilities retain input order for ties.
    """
    scene_id: str
    hypotheses: tuple[Hypothesis, ...]
    created_timestamp: float
    normalization_method: str = 'active_sum'

    def __post_init__(self):
        identifier(self.scene_id, 'scene_id')
        timestamps(self, ('created_timestamp',))
        if self.normalization_method != 'active_sum':
            raise ValueError('normalization_method must be active_sum')
        if not isinstance(self.hypotheses, (list, tuple)):
            raise ValueError('hypotheses must be an ordered sequence')
        object.__setattr__(self, 'hypotheses', tuple(self.hypotheses))
        seen = set()
        for item in self.hypotheses:
            if not isinstance(item, Hypothesis):
                raise ValueError('hypotheses must contain Hypothesis instances')
            if item.scene_id != self.scene_id:
                raise ValueError('hypothesis.scene_id must match HypothesisSet.scene_id')
            if item.hypothesis_id in seen:
                raise ValueError(f'duplicate hypothesis_id: {item.hypothesis_id}')
            seen.add(item.hypothesis_id)

    @property
    def normalized_posteriors(self):
        active = [h for h in self.hypotheses if h.status == HypothesisStatus.ACTIVE]
        total = fsum(h.posterior_probability for h in active)
        return MappingProxyType({h.hypothesis_id: h.posterior_probability / total
                                 if total else 1.0 / len(active) for h in active})

    @property
    def highest_probability_hypothesis(self):
        return max((h for h in self.hypotheses if h.status == HypothesisStatus.ACTIVE),
                   key=lambda h: h.posterior_probability, default=None)
