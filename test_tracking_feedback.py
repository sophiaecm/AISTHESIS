import unittest
from fifth_layer.perception.tracking import ObjectTracker
from fifth_layer.perception.temporal import extract_motion_evidence
from fifth_layer.prediction_feedback import PredictionFeedback


def detection(x, track_id=None):
    item = {"class_name": "person", "box_xyxy": [x, 100, x + 20, 120], "object_id": 0}
    if track_id is not None:
        item["track_id"] = track_id
    return item


class TrackingTests(unittest.TestCase):
    def test_short_gap_and_expiration(self):
        tracker = ObjectTracker()
        first = tracker.update([detection(100)], 0, 1000, 1000)
        second = tracker.update([detection(110)], .1, 1000, 1000)
        self.assertEqual(first[0]["track_id"], second[0]["track_id"])
        self.assertEqual(tracker.update([], .2, 1000, 1000), [])
        restored = tracker.update([detection(140)], .4, 1000, 1000)
        self.assertEqual(first[0]["track_id"], restored[0]["track_id"])
        expired = tracker.update([detection(140)], 3, 1000, 1000)
        self.assertNotEqual(first[0]["track_id"], expired[0]["track_id"])

    def test_motion_does_not_switch_identity(self):
        result = extract_motion_evidence([detection(100, 1)], [detection(101, 2)], 1000, 1000, .1)
        self.assertEqual(result, [])
        result = extract_motion_evidence([detection(100)], [detection(110)], 1000, 1000, .1)
        self.assertEqual(len(result), 1)

    def test_smoothing_does_not_combine_different_tracks(self):
        from collections import Counter, deque
        from test_analysis_connections import live_functions
        env = live_functions("_motion_candidate_from_evidence", "update_smoothed_motion")
        env.update(Counter=Counter, motion_history=deque(maxlen=7), MOTION_MIN_VOTES=4)
        for track_id in range(4):
            env["update_smoothed_motion"]([dict(track_id=track_id, class_name="person",
                motion_state="moving_right", normalized_motion=.1, speed_pixels_per_second=20)])
        self.assertEqual(env["latest_stable_motion_evidence"], [])

    def test_crossing_tracks_keep_velocity_identity(self):
        tracker = ObjectTracker()
        first = tracker.update([detection(100), detection(200)], 0, 1000, 1000)
        tracker.update([detection(130), detection(170)], .1, 1000, 1000)
        crossed = tracker.update([detection(140), detection(160)], .2, 1000, 1000)
        self.assertEqual(crossed[1]["track_id"], first[0]["track_id"])
        self.assertEqual(crossed[0]["track_id"], first[1]["track_id"])


class FeedbackTests(unittest.TestCase):
    def test_due_same_track_error(self):
        feedback = PredictionFeedback()
        feedback.record(0, 1, [120, 110])
        self.assertEqual(feedback.observe(.5, [detection(110, 1)], 1000, 1000), [])
        result = feedback.observe(1, [detection(115, 1)], 1000, 1000)
        self.assertEqual(result[0]["error_pixels"], 5)
        self.assertEqual(feedback.observe(1.1, [detection(115, 1)], 1000, 1000), [])

    def test_missing_is_not_success_and_history_bounded(self):
        feedback = PredictionFeedback(capacity=2)
        feedback.record(0, 1, [120, 110])
        self.assertEqual(feedback.observe(1, [detection(110, 2)], 1000, 1000), [])
        self.assertEqual(feedback.observe(2, [], 1000, 1000)[0]["status"], "expired")
        for index in range(10):
            feedback.record(index, index, [0, 0])
        self.assertEqual(len(feedback.pending), 2)
        feedback.reset()
        self.assertFalse(feedback.pending)
        self.assertFalse(feedback.history)


if __name__ == "__main__":
    unittest.main()
