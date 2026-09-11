"""Non-neural compact physical summaries and a model-independent feature view."""
from dataclasses import dataclass
import json

from ._structured import freeze, identifier, number
from .physical_state import plain


@dataclass(frozen=True)
class LatentPhysicalState:
    latent_state_id: str
    scene_id: str
    timestamp: float | None
    session_id: str
    coordinate_frame_id: str
    objects: tuple
    relations: tuple
    dynamics: tuple
    active_constraints: tuple
    uncertainty: tuple[str, ...]
    provenance: object
    schema_version: str = 'latent-physical-state-0.1'

    def __post_init__(self):
        for name in ('latent_state_id', 'scene_id', 'session_id', 'coordinate_frame_id'):
            identifier(getattr(self, name), name)
        if self.timestamp is not None:
            number(self.timestamp, 'timestamp', nonnegative=True)
        # Nested records are immutable mappings, not separate mutable model trees.
        for name in ('objects', 'relations', 'dynamics', 'active_constraints', 'uncertainty', 'provenance'):
            object.__setattr__(self, name, freeze(getattr(self, name)))
        ids = [o['physical_object_id'] for o in self.objects]
        if len(ids) != len(set(ids)):
            raise ValueError('duplicate latent object')
        if any(r['subject_id'] not in ids or r['object_id'] not in ids for r in self.relations):
            raise ValueError('latent relations require current physical objects')

    def to_dict(self):
        return plain(self)

    def to_json(self):
        return json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'), allow_nan=False)

    def to_feature_dict(self):
        """Structured features, not embeddings; masks never turn unknown into zero.

        Counts count explicit signals only, not all actual events in the world.
        Categorical fields retain source status, numerical fields include masks.
        """
        objects = []
        for obj in self.objects:
            numeric = {}
            categorical = {}
            for name, attr in obj['attributes'].items():
                if name in ('center', 'bbox', 'size', 'displacement', 'velocity'):
                    width = 4 if name == 'bbox' else 2
                    valid = attr['value'] is not None and attr['status'] in ('observed', 'estimated')
                    numeric[name] = {'value': plain(attr['value']), 'validity_mask': [valid] * width,
                                     'units': attr['units'], 'status': attr['status']}
                else:
                    categorical[name] = {'value': plain(attr['value']), 'status': attr['status']}
            objects.append({'physical_object_id': obj['physical_object_id'], 'numeric': numeric,
                            'categorical': categorical, 'uncertainty': list(obj['uncertainty'])})
        return {'schema_version': 'latent-physical-features-0.1', 'source_latent_state_id': self.latent_state_id,
                'scene_id': self.scene_id, 'timestamp': self.timestamp, 'coordinate_frame_id': self.coordinate_frame_id,
                'count_semantics': 'explicit source representations/signals, not confirmed physical events',
                'counts': {'object_count': len(objects),
                    'moving_object_count': sum(o['attributes']['motion_state']['value'] == 'moving' for o in self.objects),
                    'occlusion_possible_count': sum(o['attributes']['occlusion_state']['value'] == 'possible' for o in self.objects),
                    'collision_possible_count': sum(r['signal'] == 'collision_risk_geometry' for r in self.relations),
                    'support_candidate_count': sum(r['signal'] == 'support_candidate' for r in self.relations),
                    'kinematic_discontinuity_count': len({oid for d in self.dynamics if d['signal'] == 'tracking_or_motion_discontinuity'
                                                        for oid in d['object_ids']})},
                'objects': objects, 'uncertainty': list(self.uncertainty)}
