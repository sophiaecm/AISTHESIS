"""Immutable audit contracts for image-plane constraints, never physical facts."""
from dataclasses import dataclass, field
from collections.abc import Mapping
import json

from ._structured import freeze, identifier, number
from .physical_state import plain


@dataclass(frozen=True)
class PhysicsConstraintResult:
    constraint_id: str
    constraint_type: str
    scene_id: str
    timestamp: float | None
    object_ids: tuple[str, ...]
    status: str
    finding: str
    explanation: str
    measured_values: Mapping = field(default_factory=dict)
    expected_range: Mapping = field(default_factory=dict)
    derived_from: tuple[str, ...] = ()
    provenance: Mapping = field(default_factory=dict)
    uncertainty: tuple[str, ...] = ()
    severity: str = 'info'
    confidence: float | None = None
    rule_version: str = 'physics-constraints-0.2'

    def __post_init__(self):
        for name in ('constraint_id', 'constraint_type', 'scene_id', 'finding', 'explanation', 'rule_version'):
            identifier(getattr(self, name), name)
        if self.timestamp is not None:
            number(self.timestamp, 'timestamp', nonnegative=True)
        if self.status not in ('satisfied', 'violated', 'indeterminate', 'unsupported', 'not_applicable'):
            raise ValueError('invalid constraint status')
        if self.severity not in ('info', 'review'):
            raise ValueError('invalid severity')
        if self.confidence is not None:
            raise ValueError('v0.2 does not assign constraint probabilities')
        if not self.derived_from or not self.provenance:
            raise ValueError('constraint results require field references and provenance')
        for name in ('object_ids', 'derived_from', 'uncertainty'):
            values = tuple(sorted(set(getattr(self, name))))
            for value in values:
                identifier(value, name)
            object.__setattr__(self, name, values)
        for name in ('measured_values', 'expected_range', 'provenance'):
            if not isinstance(getattr(self, name), Mapping):
                raise ValueError(f'{name} must be a mapping')
            object.__setattr__(self, name, freeze(getattr(self, name)))


@dataclass(frozen=True)
class PhysicsConstraintBundle:
    scene_id: str
    timestamp: float | None
    results: tuple[PhysicsConstraintResult, ...] = ()
    schema_version: str = 'physics-constraints-0.2'

    def __post_init__(self):
        identifier(self.scene_id, 'scene_id')
        if self.timestamp is not None:
            number(self.timestamp, 'timestamp', nonnegative=True)
        results = tuple(sorted(self.results, key=lambda r: (r.constraint_type, r.object_ids, r.constraint_id)))
        if any(r.scene_id != self.scene_id or r.timestamp != self.timestamp for r in results):
            raise ValueError('constraint result must match bundle scene and time')
        if len({r.constraint_id for r in results}) != len(results):
            raise ValueError('duplicate constraint ID')
        object.__setattr__(self, 'results', results)


@dataclass(frozen=True)
class PhysicsTransitionAssessment:
    assessment_id: str
    scene_id: str
    timestamp: float | None
    previous_scene_id: str | None
    previous_timestamp: float | None
    constraints: PhysicsConstraintBundle
    provenance: Mapping
    assessment_kind: str = 'retrospective_transition_ending_at_current_snapshot'
    schema_version: str = 'physics-transition-0.2'

    def __post_init__(self):
        identifier(self.assessment_id, 'assessment_id')
        identifier(self.scene_id, 'scene_id')
        identifier(self.previous_scene_id, 'previous_scene_id', optional=True)
        for name in ('timestamp', 'previous_timestamp'):
            if getattr(self, name) is not None:
                number(getattr(self, name), name, nonnegative=True)
        if self.timestamp is not None and self.previous_timestamp is not None and self.previous_timestamp >= self.timestamp:
            raise ValueError('transition history must precede current timestamp')
        if self.constraints.scene_id != self.scene_id or self.constraints.timestamp != self.timestamp:
            raise ValueError('assessment and bundle must agree')
        object.__setattr__(self, 'provenance', freeze(self.provenance))

    def to_dict(self):
        return plain(self)

    def to_json(self):
        return json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'), allow_nan=False)
