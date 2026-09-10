"""Opt-in conversion from real legacy WorldState fields, without invoking reasoners."""
from collections.abc import Mapping
from uuid import uuid4

from fifth_layer.world_state import WorldState
from ._structured import SCHEMA_VERSION, freeze
from .scene_state import SceneState


class WorldStateAdapter:
    """Missing source fields stay absent/empty; no risk or evidence is inferred.

    WorldState has no native ID. Caller-supplied source_world_state_id and
    snapshot_sequence_id can link external snapshot/job metadata. scene_id is
    generated if omitted; it is never presented as a legacy WorldState ID.
    Reasoner-derived fields are copied only if explicitly present in data.
    """
    @staticmethod
    def from_world_state(world_state: WorldState, *, scene_id=None,
                         source_world_state_id=None, snapshot_sequence_id=None,
                         provenance=None):
        if not isinstance(world_state, WorldState):
            raise ValueError('world_state must be a WorldState')
        if not isinstance(world_state.data, Mapping):
            raise ValueError('WorldState.data must be a mapping')
        # Project first: no recursive traversal of unrelated image payloads.
        data = world_state.data
        mapping = {'timestamp': 'WorldState.timestamp'}

        def take(source, destination, default=None):
            if source not in data or data[source] is None:
                return default
            mapping[destination] = f'WorldState.data.{source}'
            return freeze(data[source], source, sanitize=True)

        def group(sources, destination):
            result = {}
            for source in sources:
                value = take(source, f'{destination}.{source}')
                if value is not None:
                    result[source] = value
            return result

        observed_source = 'accepted_detections' if 'accepted_detections' in data else 'detections'
        observed = take(observed_source, 'observed_objects', ()) or ()
        # Legacy unlabelled detector records are observed. Explicitly missing,
        # stale or predicted records are never promoted to observations.
        observed = tuple(item for item in observed if isinstance(item, Mapping)
                         and not item.get('is_predicted')
                         and item.get('observation_state', 'observed') == 'observed')
        values = dict(
            observed_objects=observed,
            predicted_tracks=take('predicted_tracks', 'predicted_tracks', ()) or (),
            image_width=take('image_width', 'image_width'),
            image_height=take('image_height', 'image_height'),
            spatial_relations=take('scene_relations', 'spatial_relations', ()) or (),
            motion_evidence=take('motion_evidence', 'motion_evidence', ()) or (),
            occlusion_evidence=group(('occlusion_evidence', 'occlusion_reasoning'), 'occlusion_evidence'),
            physics_evidence=group(('position', 'velocity', 'dt', 'occlusion_zone',
                                    'physics_hidden_interaction_possible', 'physics_confidence'), 'physics_evidence'),
            latent_evidence=group(('latent_occlusion_state', 'latent_temporal_state',
                                  'latent_scene_state', 'latent_hypothesis', 'latent_physical_risk',
                                  'latent_risk', 'auditory_latent_state'), 'latent_evidence'),
            semantic_evidence=group(('semantic_evidence', 'scene_description'), 'semantic_evidence'),
            risk=take('risk_level', 'risk'),
            uncertainty=take('uncertainty', 'uncertainty'),
        )
        if values['uncertainty'] is None:
            values['uncertainty'] = take('fused_uncertainty', 'uncertainty')
        source_timestamps = {'WorldState.timestamp': world_state.timestamp}
        for key in ('observation_timestamp', 'snapshot_timestamp', 'fast_scene_timestamp'):
            if key in data:
                source_timestamps[f'WorldState.data.{key}'] = freeze(data[key], sanitize=True)
        metadata = group(('source_type', 'source_path', 'model', 'perception_sources'), 'provenance.source_metadata')
        supplied = freeze(provenance or {}, 'provenance', sanitize=True)
        if not isinstance(supplied, Mapping):
            raise ValueError('provenance must be a structured mapping')
        retained_provenance = dict(
            schema_version=SCHEMA_VERSION,
            source_component='fifth_layer.world_state.WorldState',
            adapter_component='fifth_layer.world_model.adapter.WorldStateAdapter',
            source_field_mapping=mapping, source_timestamps=source_timestamps,
            source_metadata=metadata, supplied=supplied,
            observation_filter='exclude explicitly predicted or non-observed detections',
        )
        if source_world_state_id is not None:
            retained_provenance['source_world_state_id'] = source_world_state_id
        if snapshot_sequence_id is not None:
            retained_provenance['snapshot_sequence_id'] = snapshot_sequence_id
        return SceneState(scene_id=scene_id if scene_id is not None else str(uuid4()),
                          timestamp=world_state.timestamp,
                          source_world_state_id=source_world_state_id,
                          snapshot_sequence_id=snapshot_sequence_id,
                          provenance=retained_provenance, **values)
