"""Track lifecycle and live integration tests, without model loading."""
import unittest
from unittest.mock import Mock
from fifth_layer.perception.tracking import TrackMemory
from fifth_layer.perception.temporal import extract_motion_evidence
from fifth_layer.world_state import WorldState
from test_analysis_connections import live_functions


def detection(x=100, confidence=.87, **extra):
    return dict(class_name='person', box_xyxy=[x, 100, x+20, 120],
                confidence=confidence, **extra)


class TrackMemoryTests(unittest.TestCase):
    def setUp(self):
        self.memory = TrackMemory()

    def update(self, items, timestamp):
        return self.memory.update(items, timestamp, 1000, 1000)

    def test_small_motion_and_fields(self):
        original = detection()
        first = self.update([original], 0)[0]['track_id']
        second = self.update([detection(105)], .1)[0]['track_id']
        self.assertIsInstance(first, int)
        self.assertEqual(first, second)
        self.assertNotIn('track_id', original)
        track = self.memory.tracks[first]
        self.assertEqual(track['bbox'], (105, 100, 125, 120))
        self.assertEqual(track['center'], (115, 110))
        self.assertEqual(track['confidence'], .87)
        self.assertEqual((track['velocity_x'], track['velocity_y']), (50, 0))
        self.assertEqual(track['first_seen_timestamp'], 0)
        self.assertEqual(track['last_seen_timestamp'], .1)
        self.assertEqual(track['age_frames'], 2)
        self.assertEqual(track['missed_frames'], 0)
        self.assertEqual(track['status'], 'active')

    def test_two_objects_and_reordered_input(self):
        first = self.update([detection(100), detection(500)], 0)
        second = self.update([detection(505), detection(105)], .1)
        self.assertNotEqual(first[0]['track_id'], first[1]['track_id'])
        self.assertEqual([d['track_id'] for d in second],
                         [first[1]['track_id'], first[0]['track_id']])

    def test_missing_and_reappearance(self):
        identity = self.update([detection()], 0)[0]['track_id']
        self.update([detection(110)], .1)
        for t in (.2, .3, .4):
            self.assertEqual(self.update([], t), [])
        track = self.memory.tracks[identity]
        self.assertEqual(track['status'], 'temporarily_missing')
        self.assertEqual(track['missed_frames'], 3)
        self.assertEqual(track['bbox'], (110, 100, 130, 120))
        self.assertEqual(track['last_seen_timestamp'], .1)
        self.assertEqual(self.update([detection(140)], .5)[0]['track_id'], identity)
        self.assertEqual(self.memory.tracks[identity]['status'], 'active')
        self.assertEqual(self.memory.tracks[identity]['missed_frames'], 0)

    def test_timeout_expires_before_matching(self):
        identity = self.update([detection()], 0)[0]['track_id']
        new = self.update([detection()], 1.51)[0]['track_id']
        self.assertNotEqual(identity, new)
        self.assertNotIn(identity, self.memory.tracks)
        self.assertEqual(self.memory.expired_tracks[0]['status'], 'expired')
        self.update([], 1.6)
        self.assertEqual(self.memory.expired_tracks, [])

    def test_missed_frame_limit(self):
        self.memory = TrackMemory(max_missed_frames=2)
        identity = self.update([detection()], 0)[0]['track_id']
        self.update([], .1)
        self.update([], .2)
        self.assertIn(identity, self.memory.tracks)
        self.update([], .3)
        self.assertNotIn(identity, self.memory.tracks)
        self.assertEqual(self.memory.expired_tracks[0]['status'], 'expired')

    def test_incompatible_class_and_distance(self):
        identity = self.update([detection()], 0)[0]['track_id']
        different = detection()
        different['class_name'] = 'chair'
        result = self.update([different, detection(800)], .1)
        self.assertTrue(all(d['track_id'] != identity for d in result))

    def test_iou_breaks_equal_center_tie(self):
        wide = detection()
        wide['box_xyxy'] = [90, 90, 130, 130]
        first = self.update([detection(), wide], 0)
        second = self.update([wide, detection()], .1)
        self.assertEqual(second[0]['track_id'], first[1]['track_id'])

    def test_legacy_box_and_motion(self):
        legacy = dict(class_name='person', box=[100, 100, 20, 20])
        self.assertIsInstance(self.update([legacy], 0)[0]['track_id'], int)
        evidence = extract_motion_evidence([legacy], [detection(110)], 1000, 1000, .1)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]['velocity_x'], 100)

    def test_equal_timestamp_and_backwards_validation(self):
        identity = self.update([detection()], 1)[0]['track_id']
        self.assertEqual(self.update([detection()], 1)[0]['track_id'], identity)
        with self.assertRaises(ValueError):
            self.update([], .5)
        self.assertIn(identity, self.memory.tracks)

    def test_live_world_and_label(self):
        env = live_functions('track_observation', 'filter_live_tracking_detections',
                             'draw_tracked_detections')
        env.update(object_tracker=self.memory, LIVE_MIN_TRACK_CONFIDENCE=.65,
                   prediction_feedback=Mock(), cv2=Mock())
        state = WorldState(0, dict(image_width=1000, image_height=1000,
                                  detections=[detection(), detection(500, .4)]))
        visible = env['track_observation'](state)
        self.assertEqual(len(visible), 1)
        self.assertTrue(all(isinstance(d['track_id'], int) for d in state.data['detections']))
        env['draw_tracked_detections'](Mock(), state.data['detections'])
        labels = [call.args[1] for call in env['cv2'].putText.call_args_list]
        self.assertEqual(labels, ['person #0 0.87', 'person #1 0.40'])


if __name__ == '__main__':
    unittest.main()
