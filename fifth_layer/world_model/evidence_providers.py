"""Read-only projections of existing outputs. No reasoner/model calls."""
from collections.abc import Mapping

from fifth_layer.expected_consequences import ExpectedConsequences
from fifth_layer.future_state import FutureState
from fifth_layer.latent_state import LatentState
from ._structured import freeze
from .evidence import EvidenceItem, EvidenceBundle, stable_id


MOVING = frozenset({'moving', 'moving_right', 'moving_left', 'moving_up', 'moving_down'})


def _output(output):
    if isinstance(output, ExpectedConsequences):
        return output.predictions, 'ExpectedConsequences.predictions'
    if isinstance(output, FutureState):
        return output.data, 'FutureState.data'
    if isinstance(output, LatentState):
        return output.features, 'LatentState.features'
    if isinstance(output, Mapping):
        return output, 'mapping'
    raise ValueError('output must be ExpectedConsequences, FutureState, LatentState or a mapping')


def _motion_links(state):
    if state in MOVING:
        return ('continued_motion',), ('object_stops',)
    if state == 'stationary':
        return ('object_stops',), ('continued_motion',)
    return (), ()


def _item(scene, category, component, field, value, *, supports=(), contradicts=(),
          timestamp=None, origin=None):
    value = freeze(value, sanitize=True)
    if not isinstance(value, Mapping):
        raise ValueError('provider value must be a mapping')
    # None means unknown, never manufactured confidence or probability.
    confidence = value.get('confidence', value.get('raw_confidence', value.get('physics_confidence')))
    stamp = value.get('timestamp', value.get('observation_timestamp',
                      timestamp if timestamp is not None else scene.timestamp))
    provenance = dict(source_field=origin or field, source_timestamp=stamp,
                      source_fields=tuple(f'{origin or field}.{key}' for key in sorted(value)),
                      adapter_version='0.2', scene_provenance=scene.provenance,
                      origin=origin or field,
                      confidence_policy='source confidence only; None when absent')
    identity = stable_id('e', scene.scene_id, category, component, field, value, stamp,
                         supports, contradicts, provenance)
    return EvidenceItem(identity, scene.scene_id, category, component, field, value, stamp,
                        confidence, supports, contradicts, value.get('track_id'),
                        value.get('object_id', value.get('current_object_id')), provenance)


class PhysicsEvidenceProvider:
    component = 'fifth_layer.reasoners.physics.PhysicsReasoner'

    def provide(self, scene, output=None, *, timestamp=None):
        data, origin = (scene.physics_evidence, 'SceneState.physics_evidence') if output is None else _output(output)
        fields = ('position', 'velocity', 'dt', 'occlusion_zone', 'expected_next_position',
                  'enters_occlusion_zone', 'hidden_interaction_possible',
                  'physics_hidden_interaction_possible', 'physics_confidence',
                  'latent_physical_risk', 'track_id', 'object_id', 'timestamp', 'observation_timestamp')
        value = {key: data[key] for key in fields if key in data}
        if not value:
            return ()
        supports = ('object_becomes_occluded',) if data.get('enters_occlusion_zone') is True else ()
        # Generic hidden interaction says nothing about a new actor.
        return (_item(scene, 'physics', self.component, 'physics_output', value,
                      supports=supports, timestamp=timestamp, origin=origin),)


class TemporalEvidenceProvider:
    component = 'fifth_layer.reasoners.temporal_prediction.TemporalPredictionReasoner'

    def provide(self, scene, output=None, *, timestamp=None):
        if output is None:
            return ()
        data, origin = _output(output)
        fields = ('track_id', 'motion_state', 'current_center', 'velocity_x', 'velocity_y',
                  'trajectory', 'trajectory_available', 'prediction_basis', 'expected_motion_continuation',
                  'occlusion_prediction', 'raw_confidence', 'predicted_event', 'timestamp', 'observation_timestamp')
        value = {key: data[key] for key in fields if key in data}
        if not value:
            return ()
        supports, contradicts = _motion_links(data.get('motion_state'))
        if data.get('occlusion_prediction') in ('approaching_occlusion', 'occlusion_interaction_possible'):
            supports += ('object_becomes_occluded',)
        # no_predicted_occlusion is not an observed negative, so adds no contradiction.
        return (_item(scene, 'temporal', self.component, 'temporal_output', value,
                      supports=supports, contradicts=contradicts, timestamp=timestamp, origin=origin),)


class OcclusionEvidenceProvider:
    component = 'fifth_layer.reasoners.occlusion.OcclusionReasoner'

    def provide(self, scene, output=None, *, timestamp=None):
        if output is not None:
            data, origin = _output(output)
            records = data.get('occlusion_hypotheses', ())
            field = 'occlusion_hypotheses'
        else:
            data, origin = scene.occlusion_evidence, 'SceneState.occlusion_evidence'
            records = data.get('occlusion_evidence', ()) if isinstance(data, Mapping) else data
            field = 'occlusion_evidence'
        result = []
        for record in records or ():
            possible = (record.get('has_overlap_evidence') is True or
                        record.get('hypothesis') in ('occlusion_possible', 'occlusion_likely'))
            # Frame truncation alone is insufficient for an occluded-object future.
            if record.get('frame_truncated') and not record.get('has_overlap_evidence'):
                possible = False
            result.append(_item(scene, 'occlusion', self.component if output is not None else
                               'fifth_layer.perception.occlusion.extract_occlusion_evidence',
                               field, record, supports=('object_becomes_occluded',) if possible else (),
                               timestamp=timestamp, origin=f'{origin}.{field}[*]'))
        if output is None and isinstance(data, Mapping):
            expected = data.get('occlusion_reasoning', {}).get('expected', {})
            for record in expected.get('occlusion_hypotheses', ()):
                possible = record.get('hypothesis') in ('occlusion_possible', 'occlusion_likely')
                result.append(_item(scene, 'occlusion', self.component, 'occlusion_hypotheses', record,
                    supports=('object_becomes_occluded',) if possible else (), timestamp=timestamp,
                    origin='SceneState.occlusion_evidence.occlusion_reasoning.expected.occlusion_hypotheses[*]'))
        return tuple(result)


class SemanticEvidenceProvider:
    def provide(self, scene, output=None, *, timestamp=None):
        if output is None:
            data = scene.semantic_evidence.get('semantic_evidence', {})
            origin = 'SceneState.semantic_evidence.semantic_evidence'
        else:
            data, origin = _output(output)
        result = []
        fields = ('description_available', 'mentioned_objects', 'mentioned_actions',
                  'mentioned_relations', 'semantic_text')
        value = {key: data[key] for key in fields if key in data}
        if value:
            # Unbound text mentions do not identify actors or physical trajectories.
            result.append(_item(scene, 'semantic', 'fifth_layer.perception.semantic_evidence.extract_semantic_evidence',
                                'semantic_evidence', value, timestamp=timestamp, origin=origin))
        for record in data.get('semantic_conflicts', ()):
            # The legacy reasoner reports description omissions as conflicts.
            # Preserve its report, but never reinterpret omissions as contradiction.
            result.append(_item(scene, 'semantic', 'fifth_layer.reasoners.semantic_conflict.SemanticConflictReasoner',
                                'semantic_conflicts', record, timestamp=timestamp,
                                origin=f'{origin}.semantic_conflicts[*]'))
        return tuple(result)


class MotionEvidenceProvider:
    def provide(self, scene):
        return tuple(_item(scene, 'motion', 'fifth_layer.perception.temporal.extract_motion_evidence',
                           'motion_evidence', record, supports=_motion_links(record.get('motion_state'))[0],
                           contradicts=_motion_links(record.get('motion_state'))[1],
                           origin='SceneState.motion_evidence[*]')
                     for record in scene.motion_evidence)


class TrackingEvidenceProvider:
    def provide(self, scene):
        result = []
        for field in ('observed_objects', 'predicted_tracks'):
            for record in getattr(scene, field):
                value = {key: record[key] for key in (
                    'track_id', 'object_id', 'class_name', 'observation_state', 'is_predicted',
                    'confidence', 'raw_confidence', 'predicted_center', 'predicted_bbox',
                    'possible_occlusion', 'position_uncertainty', 'position_uncertainty_units',
                    'timestamp', 'last_seen_timestamp', 'prediction_valid_until') if key in record}
                supports = ('object_reappears',) if (field == 'predicted_tracks' and
                           record.get('possible_occlusion') is True and record.get('track_id') is not None
                           and record.get('is_predicted') is True
                           and record.get('observation_state') == 'predicted') else ()
                result.append(_item(scene, 'tracking', 'fifth_layer.perception.tracking.TrackMemory',
                                    field, value, supports=supports, origin=f'SceneState.{field}[*]'))
        return tuple(result)


def collect_evidence(scene, *, physics=None, temporal=None, occlusion=None, semantic=None,
                     source_timestamps=None):
    """Read scene summaries plus explicitly supplied outputs, never compute outputs.

    source_timestamps supplies observation timestamps for outputs that carry none.
    Identical duplicate evidence IDs are merged, with no confidence accumulation.
    """
    source_timestamps = source_timestamps or {}
    items = []
    for name, provider, output in (
        ('physics', PhysicsEvidenceProvider(), physics),
        ('temporal', TemporalEvidenceProvider(), temporal),
        ('occlusion', OcclusionEvidenceProvider(), occlusion),
        ('semantic', SemanticEvidenceProvider(), semantic)):
        items.extend(provider.provide(scene, output, timestamp=source_timestamps.get(name)))
    items.extend(MotionEvidenceProvider().provide(scene))
    items.extend(TrackingEvidenceProvider().provide(scene))
    unique = {item.evidence_id: item for item in items}
    return EvidenceBundle(scene.scene_id, tuple(unique.values()),
                          {'provider_version': '0.2', 'missing_evidence_is_contradiction': False})
