"""Regressions for articulation and sparse live YOLO observations."""
import unittest
from fifth_layer.perception.tracking import TrackMemory


def person(box, confidence=.9):
    return dict(class_name='person', box_xyxy=box, confidence=confidence)


class LiveIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tracker = TrackMemory(preserve_observation_continuity=True, debug=True)

    def update(self, boxes, timestamp):
        return self.tracker.update([person(box) for box in boxes], timestamp, 640, 480)

    def test_extended_arm_expands_box(self):
        first = self.update([[100, 50, 300, 450]], 0)
        second = self.update([[100, 50, 500, 450]], .1)
        self.assertEqual(first[0]['track_id'], second[0]['track_id'])
        record = self.tracker.match_debug[0]
        self.assertEqual(record['iou'], .5)
        self.assertGreater(record['center_distance'], 800 * .12)
        self.assertEqual(record['reason'], 'accepted_overlap')
        print('\nArticulation diagnostic:', record)

    def test_high_iou_translation_beyond_gate(self):
        first = self.update([[0, 0, 400, 450]], 0)
        second = self.update([[100, 0, 500, 450]], .1)
        self.assertEqual(first[0]['track_id'], second[0]['track_id'])
        self.assertAlmostEqual(self.tracker.match_debug[0]['iou'], .6)

    def test_long_inference_gap_visible_person(self):
        first = self.update([[100, 50, 300, 450]], 0)
        for timestamp in (3, 7, 12):
            result = self.update([[100, 50, 300, 450]], timestamp)
            self.assertEqual(first[0]['track_id'], result[0]['track_id'])
            self.assertEqual(self.tracker.tracks[first[0]['track_id']]['missed_frames'], 0)
        print('\nSlow inference diagnostic:', self.tracker.match_debug[0])

    def test_two_neighbors_remain_separate(self):
        first = self.update([[50, 50, 200, 450], [300, 50, 450, 450]], 0)
        second = self.update([[300, 50, 450, 450], [50, 50, 280, 450]], .1)
        self.assertEqual([d['track_id'] for d in second],
                         [first[1]['track_id'], first[0]['track_id']])
        self.assertEqual(len(set(d['track_id'] for d in second)), 2)

    def test_crossing_retains_velocity_identity(self):
        def boxes(a, b):
            return [[a, 100, a+20, 120], [b, 100, b+20, 120]]
        first = self.update(boxes(100, 200), 0)
        self.update(boxes(130, 170), .1)
        crossed = self.update(boxes(140, 160), .2)
        self.assertEqual([d['track_id'] for d in crossed],
                         [first[1]['track_id'], first[0]['track_id']])

    def test_real_missing_timeout_gets_new_id(self):
        first = self.update([[100, 50, 300, 450]], 0)
        self.update([], .1)
        returned = self.update([[100, 50, 300, 450]], 3)
        self.assertNotEqual(first[0]['track_id'], returned[0]['track_id'])
        self.assertEqual(self.tracker.expired_tracks[0]['status'], 'expired')
        self.assertEqual(self.tracker.match_debug[0]['reason'], 'missing_timeout')
        print('\nMissing diagnostic:', self.tracker.match_debug[0])

    def test_confidence_change_does_not_change_id(self):
        a = self.tracker.update([person([100, 50, 300, 450])], 0, 640, 480)
        b = self.tracker.update([person([100, 50, 300, 450], .3)], .1, 640, 480)
        self.assertEqual(a[0]['track_id'], b[0]['track_id'])


if __name__ == '__main__':
    unittest.main()
