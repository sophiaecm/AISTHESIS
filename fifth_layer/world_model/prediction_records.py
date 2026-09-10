"""Immutable observational-loop records. No inference, scoring updates or I/O."""
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isclose

from ._structured import freeze_fields, geometry, identifier, number, timestamps
from .experience_memory import EvaluationStatus


def _identity(instance):
    for name in ('track_id', 'object_id'):
        value = getattr(instance, name)
        if value is not None and type(value) not in (str, int):
            raise ValueError(f'{name} must be a string, integer or None')
    size = instance.image_size
    if size is not None:
        if not isinstance(size, (list, tuple)) or len(size) != 2:
            raise ValueError('image_size must contain width and height')
        for value in size:
            number(value, 'image_size', nonnegative=True)
            if value == 0:
                raise ValueError('image_size must be positive')


def _state(value, observed=False):
    if not isinstance(value, Mapping):
        raise ValueError('state must be a structured mapping')
    if value.get('center') is not None:
        geometry(value['center'], 'center', 2)
    if value.get('visibility') not in (None, 'visible', 'occluded'):
        raise ValueError('visibility must be visible, occluded or unknown')
    if observed:
        if (value.get('is_predicted') or value.get('observed', True) is not True
                or value.get('epistemic_status', 'observed') != 'observed'):
            raise ValueError('predicted/inferred state is not an observed outcome')
        events = value.get('events', {})
        if not isinstance(events, Mapping) or any(type(v) is not bool for v in events.values()):
            raise ValueError('observed events must map explicit event names to booleans')


@dataclass(frozen=True)
class PredictionRecord:
    prediction_id: str
    scene_id: str
    hypothesis_id: str
    hypothesis_type: str
    source_timestamp: float
    target_timestamp: float
    horizon_seconds: float
    predicted_state: Mapping = field(default_factory=dict)
    track_id: int | str | None = None
    object_id: int | str | None = None
    trajectory_id: str | None = None
    image_size: tuple | None = None
    confidence: float | None = None
    uncertainty: float | None = None
    evidence_for: tuple[str, ...] = ()
    evidence_against: tuple[str, ...] = ()
    provenance: Mapping = field(default_factory=dict)

    def __post_init__(self):
        for name in ('prediction_id', 'scene_id', 'hypothesis_id', 'hypothesis_type'):
            identifier(getattr(self, name), name)
        identifier(self.trajectory_id, 'trajectory_id', optional=True)
        timestamps(self, ('source_timestamp', 'target_timestamp', 'horizon_seconds'))
        if not isclose(self.source_timestamp + self.horizon_seconds, self.target_timestamp,
                       rel_tol=0., abs_tol=1e-9):
            raise ValueError('target_timestamp must equal source_timestamp + horizon_seconds')
        _identity(self)
        for name in ('confidence', 'uncertainty'):
            if getattr(self, name) is not None:
                number(getattr(self, name), name, unit=True)
        freeze_fields(self, ('predicted_state', 'image_size', 'evidence_for', 'evidence_against', 'provenance'))
        _state(self.predicted_state)
        for name in ('evidence_for', 'evidence_against'):
            if not isinstance(getattr(self, name), tuple):
                raise ValueError(f'{name} must be an ordered sequence')
            for value in getattr(self, name):
                identifier(value, name)


@dataclass(frozen=True)
class OutcomeRecord:
    outcome_id: str
    prediction_id: str
    scene_id: str | None
    observation_timestamp: float | None
    association_status: str
    observed_state: Mapping = field(default_factory=dict)
    track_id: int | str | None = None
    object_id: int | str | None = None
    image_size: tuple | None = None
    association_confidence: float | None = None
    unavailable: tuple[str, ...] = ()
    provenance: Mapping = field(default_factory=dict)

    def __post_init__(self):
        identifier(self.outcome_id, 'outcome_id')
        identifier(self.prediction_id, 'prediction_id')
        identifier(self.scene_id, 'scene_id', optional=True)
        timestamps(self, ('observation_timestamp',), optional=('observation_timestamp',))
        if self.association_status not in ('matched', 'missing', 'ambiguous', 'unbound'):
            raise ValueError('invalid association_status')
        _identity(self)
        if self.association_confidence is not None:
            number(self.association_confidence, 'association_confidence', unit=True)
        freeze_fields(self, ('observed_state', 'image_size', 'unavailable', 'provenance'))
        _state(self.observed_state, observed=True)
        if self.association_status != 'matched' and self.observed_state:
            raise ValueError('unmatched outcomes cannot assert object observations')
        if self.association_status == 'matched' and (self.scene_id is None or
                (self.track_id is None and self.object_id is None)):
            raise ValueError('matched outcome requires a source scene and explicit identity')


@dataclass(frozen=True)
class PredictionEvaluation:
    evaluation_id: str
    prediction_id: str
    outcome_id: str
    status: EvaluationStatus
    evaluated_timestamp: float | None
    metrics: Mapping = field(default_factory=dict)
    reasons: tuple[str, ...] = ()
    missing_fields: tuple[str, ...] = ()
    provenance: Mapping = field(default_factory=dict)

    def __post_init__(self):
        for name in ('evaluation_id', 'prediction_id', 'outcome_id'):
            identifier(getattr(self, name), name)
        object.__setattr__(self, 'status', EvaluationStatus(self.status))
        timestamps(self, ('evaluated_timestamp',), optional=('evaluated_timestamp',))
        freeze_fields(self, ('metrics', 'reasons', 'missing_fields', 'provenance'))
        if not isinstance(self.metrics, Mapping):
            raise ValueError('metrics must be a structured mapping')
        for name, value in self.metrics.items():
            if value is None:
                continue
            if name.endswith('_match'):
                if type(value) is not bool:
                    raise ValueError('match metrics must be boolean or None')
            elif name in ('position_error_pixels', 'normalized_position_error'):
                number(value, name, nonnegative=True)
            elif name == 'temporal_error_seconds':
                number(value, name)

    def as_prediction_error(self):
        """Use the existing legacy error container without its live evaluator."""
        from fifth_layer.prediction_error import PredictionError
        from ._structured import freeze
        return PredictionError(details=dict(freeze(self)))
