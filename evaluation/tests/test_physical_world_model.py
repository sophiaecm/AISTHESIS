"""Physical state epistemic boundaries and offline-only replay."""
from dataclasses import FrozenInstanceError, replace
import json
import unittest

from fifth_layer.world_model import SceneState, EvidenceItem, EvidenceBundle
from fifth_layer.world_model.physical_state import PhysicalAttribute
from fifth_layer.world_model.physical_state_builder import PhysicalStateBuilder
from evaluation.physical_world_model import process_saved_result


def obj(track=1, x=10, **kwargs):
    return dict(track_id=track, class_name='ball', box_xyxy=(x, 10, x+10, 20), **kwargs)


def scene(t=1, objects=None, **kwargs):
    return SceneState('s'+str(t), timestamp=t, image_width=100, image_height=100,
                      observed_objects=(obj(),) if objects is None else objects, **kwargs)


def evidence(s, source='semantic', kind='claim', value=None, **kwargs):
    return EvidenceBundle(s.scene_id, (EvidenceItem('e', s.scene_id, source, 'test', kind,
        value or {'text': 'a hidden human collides with the falling ball'}, s.timestamp, **kwargs),))


class PhysicalWorldTests(unittest.TestCase):
    def build(self, s=None, **kwargs):
        return PhysicalStateBuilder().build(s or scene(), **kwargs)

    def attrs(self, s=None, **kwargs):
        return self.build(s, **kwargs).objects[0].attributes

    def test_empty(self):
        state = self.build(scene(objects=()))
        self.assertEqual(state.objects, ())
        self.assertEqual(state.relations, ())

    def test_stationary_source_classification(self):
        a = self.attrs(scene(motion_evidence=({'track_id': 1, 'dx': .01, 'dy': 0, 'motion_state': 'stationary'},)))
        self.assertEqual(a['motion_state'].value, 'stationary')
        self.assertEqual(a['displacement'].value, (.01, 0))
        self.assertIsNone(a['velocity'].value)

    def test_tracked_moving(self):
        a = self.attrs(scene(2, (obj(x=20),)), previous_scene=scene(1))
        self.assertEqual(a['displacement'].value, (10, 0))
        self.assertEqual(a['velocity'].value, (10, 0))
        self.assertEqual(a['velocity'].units, 'pixels/second')
        self.assertEqual(a['velocity'].status, 'estimated')
        self.assertEqual(len(a['velocity'].derived_from), 4)
        self.assertIsNone(a['motion_state'].value)

    def test_configured_deadband(self):
        b = PhysicalStateBuilder(minimum_motion_pixels=1)
        s = b.build(scene(2), previous_scene=scene(1))
        self.assertEqual(s.objects[0].attributes['motion_state'].value, 'stationary')
        s = b.build(scene(2, (obj(x=20),)), previous_scene=scene(1))
        self.assertEqual(s.objects[0].attributes['motion_state'].value, 'moving')

    def test_missing_delta(self):
        a = self.attrs(scene(None, (obj(x=20),), snapshot_sequence_id=2),
                       previous_scene=scene(None, snapshot_sequence_id=1))
        self.assertEqual(a['displacement'].value, (10, 0))
        self.assertIsNone(a['velocity'].value)

    def test_missing_geometry(self):
        a = self.attrs(scene(objects=({'track_id': 1},)))
        for field in ('bbox', 'center', 'size', 'velocity', 'frame_truncation'):
            self.assertIsNone(a[field].value)

    def test_explicit_motion_timing(self):
        a = self.attrs(scene(motion_evidence=({'track_id': 1, 'dx': 2, 'dy': 0, 'dt': .5},)))
        self.assertEqual(a['velocity'].value, (4, 0))

    def test_asynchronous_observation_rejected(self):
        with self.assertRaises(ValueError):
            self.build(scene(objects=(obj(timestamp=.5),)))

    def test_future_provenance_rejected(self):
        s = scene()
        with self.assertRaises(ValueError):
            self.build(s, evidence=evidence(s, 'motion', 'motion_evidence', {'track_id': 1, 'dx': 1, 'dy': 0},
                                           provenance={'source_timestamp': 2}))

    def test_overlap_not_collision_or_contact(self):
        state = self.build(scene(objects=(obj(), obj(2, 15))))
        types = {r.relation for r in state.relations}
        self.assertIn('overlaps', types)
        self.assertIn('left_of', types)
        self.assertFalse(types & {'collision', 'contact_possible', 'support_possible'})

    def test_frame_truncation(self):
        a = self.attrs(scene(objects=(obj(x=0),)))
        self.assertTrue(a['frame_truncation'].value)
        self.assertIsNone(a['occlusion_state'].value)

    def test_occlusion_does_not_create_hidden_actor(self):
        state = self.build(scene(occlusion_evidence={'occlusion_evidence': [
            {'object_id': 0, 'has_overlap_evidence': True}]}))
        self.assertEqual(len(state.objects), 1)
        self.assertEqual(state.objects[0].attributes['occlusion_state'].status, 'possible')

    def test_unknowns_and_source_confidence(self):
        a = self.attrs(scene(objects=(obj(confidence=.8),)))
        self.assertEqual(a['confidence'].value, .8)
        for field in ('depth', 'mass', 'force', 'friction', 'velocity', 'contact', 'support'):
            self.assertIsNone(a[field].value)
            self.assertEqual(a[field].status, 'unknown')

    def test_absence_is_not_disappearance(self):
        state = self.build(scene(2, (), predicted_tracks=({'track_id': 1, 'is_predicted': True},)), previous_scene=scene())
        self.assertEqual(state.objects, ())
        self.assertIn('disappearance is unknown', state.uncertainty['missing_objects'])

    def test_vlm_cannot_create_fact(self):
        s = scene()
        state = self.build(s, evidence=evidence(s))
        self.assertEqual(len(state.objects), 1)
        self.assertIsNone(state.objects[0].attributes['force'].value)
        self.assertEqual(state.provenance['used_evidence_ids'], ())

    def test_experience_cannot_create_fact(self):
        s = scene()
        a = self.attrs(s, evidence=evidence(s, 'experience', value={'mass': 4, 'velocity': [1, 1]}))
        self.assertIsNone(a['mass'].value)
        self.assertIsNone(a['velocity'].value)

    def test_unavailable_sensory_not_negative(self):
        s = scene()
        a = self.attrs(s, evidence=evidence(s, 'sensory', value={'observation_available': False},
                                           modality='tactile', epistemic_status='unavailable'))
        self.assertIsNone(a['contact'].value)

    def test_expected_sensory_not_observation(self):
        s = scene(objects=(obj(), obj(2, 30)))
        e = evidence(s, 'sensory', 'contact_observation', {'subject_object_id': 0, 'target_object_id': 1},
                     modality='tactile', epistemic_status='expected')
        self.assertNotIn('contact_possible', {r.relation for r in self.build(s, evidence=e).relations})

    def test_explicit_observed_contact_and_support(self):
        s = scene(objects=(obj(), obj(2, 30)))
        for kind in ('contact', 'support'):
            e = evidence(s, 'sensory', kind+'_observation', {'subject_object_id': 0, 'target_object_id': 1, 'observed': True},
                         modality='tactile', epistemic_status='observed')
            self.assertIn(kind+'_possible', {r.relation for r in self.build(s, evidence=e).relations})

    def test_deterministic_serialization(self):
        self.assertEqual(self.build().to_json(), self.build().to_json())
        self.assertEqual(json.loads(self.build().to_json()), self.build().to_dict())

    def test_stable_ordering(self):
        s = scene(objects=(obj(2), obj(1, 30)))
        ids = [o.object_id for o in self.build(s).objects]
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(ids, [o.object_id for o in self.build(replace(s, observed_objects=tuple(reversed(s.observed_objects)))).objects])

    def test_future_scene_rejected(self):
        with self.assertRaises(ValueError):
            self.build(scene(), previous_scene=scene(2))

    def test_future_evidence_rejected(self):
        s = scene()
        e = evidence(s)
        with self.assertRaises(ValueError):
            self.build(s, evidence=replace(e, items=(replace(e.items[0], timestamp=2),)))

    def test_nested_future_measurement_rejected(self):
        s = scene()
        with self.assertRaises(ValueError):
            self.build(s, evidence=evidence(s, 'motion', 'motion_evidence', {'track_id': 1, 'previous_timestamp': 2}))

    def test_future_object_rejected(self):
        with self.assertRaises(ValueError):
            self.build(scene(objects=(obj(timestamp=2),)))

    def test_wrong_scene_evidence_rejected(self):
        with self.assertRaises(ValueError):
            self.build(scene(), evidence=evidence(scene(2)))

    def test_immutable(self):
        s = self.build()
        with self.assertRaises(FrozenInstanceError):
            s.timestamp = 2
        with self.assertRaises(TypeError):
            s.objects[0].attributes['mass'] = PhysicalAttribute()
        with self.assertRaises(TypeError):
            s.objects[0].provenance['raw_observation']['track_id'] = 2

    def test_duplicate_track_rejected(self):
        with self.assertRaises(ValueError):
            self.build(scene(objects=(obj(), obj())))

    def test_no_untracked_history_association(self):
        a = self.attrs(scene(2, ({'center': (20, 10)},)), previous_scene=scene(1, ({'center': (10, 10)},)))
        self.assertIsNone(a['velocity'].value)

    def test_attribute_requires_provenance(self):
        with self.assertRaises(ValueError):
            PhysicalAttribute(3, 'estimated')

    def test_offline_projection_ignores_outputs_and_ground_truth(self):
        records = [dict(mode='FIFTH_LAYER_ONLY', status='ok', input={'sequence': t, 'timestamp': t, 'source_sha256': 'source'},
            fifth_layer_input={'timestamp': t, 'data': {'detections': [obj(x=10*t)]}},
            ground_truth={'human': True}, vlm_output={'hidden_human': True}) for t in (2, 1)]
        result = process_saved_result({'schema_version': 'evaluation-0.1', 'results': records})
        summary = result['sessions'][0]['summary']
        self.assertEqual(summary['snapshot_count'], 2)
        self.assertEqual(summary['object_observations_by_class'], {'ball': 2})
        self.assertEqual(summary['displacement_observations_by_class'], {'ball': 1})


if __name__ == '__main__':
    unittest.main()
