import unittest
from fifth_layer.perception.tracking import TrackMemory


def person(x=100, **extra):
    return dict(class_name='person', box_xyxy=[x, 50, x+100, 450], confidence=.9, **extra)


class ObservationLifetimeTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.tracker = TrackMemory(observation_expiration=True, min_confidence=.65,
            event_logger=lambda event, **fields: self.events.append((event, fields)))

    def update(self, detections, t):
        return self.tracker.update(detections, t, 640, 480)

    def test_slow_results_and_skipped_frames(self):
        first = self.update([person()], 0)[0]['track_id']
        # Skipped inference frames make no update calls, hence no misses.
        self.assertEqual(self.tracker.tracks[first]['missed_frames'], 0)
        self.assertEqual(self.update([person()], 30)[0]['track_id'], first)
        self.assertFalse(any(e == 'TRACK_EXPIRE' for e, _ in self.events))

    def test_false_negatives_do_not_expire(self):
        first = self.update([person()], 0)[0]['track_id']
        for t in (3, 6, 9):
            self.update([], t)
        self.assertEqual(self.tracker.tracks[first]['missed_frames'], 3)
        self.assertEqual(self.update([person()], 12)[0]['track_id'], first)

    def test_rejected_inputs_never_consume_ids(self):
        low = person()
        low['confidence'] = .4
        inputs = [low, person(accepted=False), person(uncertain=True),
                  person(predicted=True), person(status='rejected')]
        self.assertEqual(self.update(inputs, 0), [])
        self.assertEqual(self.tracker.next_id, 0)
        self.assertEqual(self.update([person()], 1)[0]['track_id'], 0)

    def test_expiration_requires_both_limits(self):
        first = self.update([person()], 0)[0]['track_id']
        for i in range(1, 10):
            self.update([], i * .01)
        self.assertIn(first, self.tracker.tracks)
        self.update([], 2)
        self.assertNotIn(first, self.tracker.tracks)
        self.assertNotEqual(self.update([person()], 3)[0]['track_id'], first)
        expired = [f for e, f in self.events if e == 'TRACK_EXPIRE']
        self.assertEqual(expired[0]['exact_expiration_reason'], 'missed_observations_and_min_elapsed')

    def test_two_people_and_diagnostics(self):
        first = self.update([person(), person(400)], 0)
        second = self.update([person(401), person(101)], 4)
        self.assertEqual([d['track_id'] for d in second], [first[1]['track_id'], first[0]['track_id']])
        self.assertEqual(len(set(d['track_id'] for d in second)), 2)
        self.assertTrue(any(e == 'TRACK_MATCH' and f['accepted_reason'].startswith('accepted_') for e, f in self.events))
        self.assertEqual(sum(e == 'TRACK_CREATE' for e, _ in self.events), 2)


if __name__ == '__main__':
    unittest.main()
