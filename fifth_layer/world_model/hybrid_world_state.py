"""Opt-in integration contracts; learned metadata never becomes physical truth."""
from collections.abc import Mapping
from dataclasses import dataclass, field
import json

from ._structured import freeze, identifier, number
from .physical_state import plain
from .latent_physical_state import LatentPhysicalState
from .physics_constraints import PhysicsConstraintBundle, PhysicsTransitionAssessment


def _metadata(value):
    # Deliberately flat and bounded: no embeddings hidden inside provenance.
    if not isinstance(value, Mapping) or len(value) > 32:
        raise ValueError('provenance requires at most 32 scalar metadata fields')
    for key, child in value.items():
        if not isinstance(key, str) or len(key) > 128:
            raise ValueError('invalid metadata key')
        if child is not None and type(child) not in (str, int, float, bool):
            raise ValueError('metadata values must be scalars, not latent payloads')
        if isinstance(child, str) and len(child) > 1024:
            raise ValueError('metadata string exceeds 1024 characters')
    return freeze(value)


def _alignment(value, physical):
    """Inspect declared source context recursively without interpreting content."""
    if isinstance(value, Mapping):
        expected = {'session_id': physical.session_id,
                    'coordinate_frame_id': physical.coordinate_frame_id}
        for key, child in value.items():
            if key in expected and child != expected[key]:
                raise ValueError(f'{key} mismatch')
            if (key == 'timestamp' or key.endswith('_timestamp')) and child is not None:
                number(child, key, nonnegative=True)
                if physical.timestamp is None or child > physical.timestamp:
                    raise ValueError('future or unorderable source timestamp')
            _alignment(child, physical)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _alignment(child, physical)


@dataclass(frozen=True)
class LearnedRepresentationSignal:
    source_model: str
    timestamp: float
    session_id: str
    scene_id: str
    representation_id: str
    feature_dimension: int
    provenance: Mapping
    representation_reference: str | None = None
    coordinate_frame_id: str | None = None
    pooled_norm: float | None = None
    temporal_change_score: float | None = None
    reference_similarity: float | None = None
    source_type: str = field(default='learned_video_representation', init=False)
    epistemic_status: str = field(default='learned_signal', init=False)
    availability: str = field(default='available', init=False)

    def __post_init__(self):
        for name in ('source_model', 'session_id', 'scene_id', 'representation_id',
                     'representation_reference', 'coordinate_frame_id'):
            value = getattr(self, name)
            identifier(value, name, optional=name in ('representation_reference', 'coordinate_frame_id'))
            if value is not None and len(value) > 1024:
                raise ValueError(f'{name} exceeds reference limit')
        number(self.timestamp, 'timestamp', nonnegative=True)
        if type(self.feature_dimension) is not int or self.feature_dimension <= 0:
            raise ValueError('feature_dimension must be a positive integer')
        for name in ('pooled_norm', 'temporal_change_score', 'reference_similarity'):
            if getattr(self, name) is not None:
                number(getattr(self, name), name, nonnegative=name == 'pooled_norm')
        if not self.provenance:
            raise ValueError('learned signal requires provenance')
        object.__setattr__(self, 'provenance', _metadata(self.provenance))


@dataclass(frozen=True)
class HybridWorldState:
    physical_state: LatentPhysicalState
    physics_constraints: PhysicsConstraintBundle | PhysicsTransitionAssessment | None = None
    learned_signal: LearnedRepresentationSignal | None = None
    timestamp: float | None = field(init=False)
    session_id: str = field(init=False)
    source_references: Mapping = field(init=False)
    availability: Mapping = field(init=False)
    uncertainty: tuple[str, ...] = field(init=False)
    provenance: Mapping = field(init=False)
    schema_version: str = field(default='hybrid-world-state-0.1', init=False)

    def __post_init__(self):
        physical = self.physical_state
        if not isinstance(physical, LatentPhysicalState) or physical.schema_version != 'latent-physical-state-0.1':
            raise ValueError('expected LatentPhysicalState v0.1')
        _alignment(plain(physical), physical)
        constraints = self.physics_constraints
        bundle = constraints
        if isinstance(constraints, PhysicsTransitionAssessment):
            if constraints.schema_version != 'physics-transition-0.2':
                raise ValueError('unsupported assessment schema')
            bundle = constraints.constraints
            for key in ('session_id', 'coordinate_frame_id'):
                if constraints.provenance.get(key) != getattr(physical, key):
                    raise ValueError(f'assessment {key} mismatch or missing')
        if bundle is not None:
            if not isinstance(bundle, PhysicsConstraintBundle) or bundle.schema_version != 'physics-constraints-0.2':
                raise ValueError('expected physics constraint bundle or assessment v0.2')
            if bundle.scene_id != physical.scene_id or bundle.timestamp != physical.timestamp:
                raise ValueError('constraints must match physical scene and timestamp')
            for result in bundle.results:
                if result.rule_version != 'physics-constraints-0.2':
                    raise ValueError('unsupported constraint rule version')
                for key in ('session_id', 'coordinate_frame_id'):
                    if result.provenance.get(key) != getattr(physical, key):
                        raise ValueError(f'constraint {key} mismatch or missing')
            _alignment(plain(constraints), physical)
        signal = self.learned_signal
        uncertainty = set(physical.uncertainty)
        if signal is not None:
            if not isinstance(signal, LearnedRepresentationSignal):
                raise ValueError('expected LearnedRepresentationSignal')
            if signal.session_id != physical.session_id or signal.scene_id != physical.scene_id:
                raise ValueError('learned session or scene mismatch')
            if physical.timestamp is None or signal.timestamp != physical.timestamp:
                raise ValueError('learned timestamp must match known physical timestamp')
            if signal.coordinate_frame_id is None:
                uncertainty.add('learned_coordinate_frame_unspecified')
            elif signal.coordinate_frame_id != physical.coordinate_frame_id:
                raise ValueError('learned coordinate frame mismatch')
            _alignment(signal.provenance, physical)
        else:
            uncertainty.add('learned_representation_unavailable')
        if constraints is None:
            uncertainty.add('physics_constraints_unavailable')
        for name in ('timestamp', 'session_id'):
            object.__setattr__(self, name, getattr(physical, name))
        object.__setattr__(self, 'source_references', freeze({
            'explicit_physical': physical.latent_state_id,
            'physics_constraint': None if bundle is None else {
                'scene_id': bundle.scene_id, 'constraint_ids': tuple(r.constraint_id for r in bundle.results),
                'assessment_id': getattr(constraints, 'assessment_id', None)},
            'learned_representation': None if signal is None else {
                'source_model': signal.source_model, 'representation_id': signal.representation_id,
                'reference': signal.representation_reference}}))
        object.__setattr__(self, 'availability', freeze({
            'explicit_physical': 'available',
            'physics_constraint': 'unavailable' if constraints is None else 'available',
            'learned_representation': 'unavailable' if signal is None else 'available'}))
        object.__setattr__(self, 'uncertainty', tuple(sorted(uncertainty)))
        object.__setattr__(self, 'provenance', freeze({
            'explicit_physical': physical.provenance,
            'physics_constraint': None if bundle is None else {
                'assessment': getattr(constraints, 'provenance', None),
                'results': {r.constraint_id: r.provenance for r in bundle.results}},
            'learned_representation': None if signal is None else signal.provenance,
            'role': 'integration_container_without_inference'}))

    def to_dict(self):
        return plain(self)

    def to_json(self):
        return json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'), allow_nan=False)
