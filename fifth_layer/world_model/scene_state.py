"""Immutable structured observation and separately labelled prediction context."""
from dataclasses import dataclass, field
from typing import Mapping, Any

from ._structured import SCHEMA_VERSION, freeze_fields, identifier, number, timestamps


@dataclass(frozen=True)
class SceneState:
    scene_id: str
    timestamp: float | None = None
    snapshot_sequence_id: int | None = None
    image_width: int | None = None
    image_height: int | None = None
    observed_objects: tuple = ()
    predicted_tracks: tuple = ()
    spatial_relations: tuple = ()
    motion_evidence: tuple = ()
    occlusion_evidence: Mapping[str, Any] = field(default_factory=dict)
    physics_evidence: Mapping[str, Any] = field(default_factory=dict)
    latent_evidence: Mapping[str, Any] = field(default_factory=dict)
    semantic_evidence: Mapping[str, Any] = field(default_factory=dict)
    risk: str | None = None
    uncertainty: float | None = None
    source_world_state_id: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        identifier(self.scene_id, 'scene_id')
        identifier(self.source_world_state_id, 'source_world_state_id', optional=True)
        identifier(self.schema_version, 'schema_version')
        if self.risk is not None:
            identifier(self.risk, 'risk')
        timestamps(self, ('timestamp',), optional=('timestamp',))
        for name in ('snapshot_sequence_id', 'image_width', 'image_height'):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f'{name} must be a non-negative integer or None')
        if self.uncertainty is not None:
            number(self.uncertainty, 'uncertainty', unit=True)
        freeze_fields(self, ('observed_objects', 'predicted_tracks', 'spatial_relations',
                            'motion_evidence', 'occlusion_evidence', 'physics_evidence',
                            'latent_evidence', 'semantic_evidence', 'provenance'))
        for name in ('observed_objects', 'predicted_tracks'):
            items = getattr(self, name)
            if not isinstance(items, tuple) or any(not isinstance(x, Mapping) for x in items):
                raise ValueError(f'{name} must be a sequence of structured object summaries')
        for item in self.observed_objects:
            if item.get('is_predicted') or item.get('observation_state', 'observed') != 'observed':
                raise ValueError('observed_objects cannot contain predicted or unobserved objects')
