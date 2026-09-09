import unittest
from unittest.mock import Mock
from fifth_layer.perception.structured_scene_narrator import FastSceneState, StructuredSceneNarrator
from test_structured_scene_narrator import world, item


class StabilityTests(unittest.TestCase):
    def setUp(self):
        self.now=0.
        self.log=Mock()
        self.fast=FastSceneState(logger=self.log,clock=lambda:self.now,wall_clock=lambda:100+self.now)

    def update(self, t, items=None, **data):
        self.now=t
        state=world(items,**data)
        state.timestamp=100+t
        self.fast.update(state)
        return self.fast.description()

    def test_confidence_jitter_counters_are_not_semantic(self):
        original=self.update(0)
        for t in (1,2,4,5):
            d=dict(item(x=151),confidence=.89,age_frames=500)
            self.assertEqual(self.update(t,[d],latency_ms=99,evaluation_count=t),original)
        events=[c for c in self.log.call_args_list if c.args[0]=='FAST_SCENE_UPDATED']
        self.assertEqual(len(events),1)

    def test_direction_needs_two_observations_and_minimum_display(self):
        def motion(direction):
            return [dict(track_id=1,motion_state=direction,normalized_motion=.1)]
        original=self.update(0,stable_motion_evidence=motion('moving_right'))
        self.assertEqual(self.update(1,stable_motion_evidence=motion('moving_left')),original)
        self.assertEqual(self.update(2,stable_motion_evidence=motion('moving_left')),original)
        self.assertIn('moving left',self.update(3.1,stable_motion_evidence=motion('moving_left')))

    def test_two_direction_observations_after_hold(self):
        self.update(0,stable_motion_evidence=[dict(track_id=1,motion_state='moving_right')])
        first=self.update(4,stable_motion_evidence=[dict(track_id=1,motion_state='moving_left')])
        self.assertIn('moving right',first)
        self.assertIn('moving left',self.update(4.1,stable_motion_evidence=[dict(track_id=1,motion_state='moving_left')]))

    def test_zone_change_requires_stability(self):
        first=self.update(0)
        self.assertEqual(self.update(4,[item(x=20)]),first)
        self.assertEqual(self.update(4.1),first)
        self.assertEqual(self.update(4.2,[item(x=20)]),first)
        self.assertIn('left',self.update(4.3,[item(x=20)]))

    def test_missing_frame_and_waiting_inference_do_not_clear(self):
        first=self.update(0)
        self.assertEqual(self.update(.1,[]),first)
        self.now=10
        self.assertEqual(self.fast.description(),first)
        self.assertEqual(self.update(10.1),first)

    def test_clear_only_after_no_track_evidence_and_hold(self):
        first=self.update(0)
        self.update(.1,[])
        self.assertEqual(self.update(.2,[]),first)
        self.now=3.1
        self.assertIn('Waiting',self.fast.description())
        self.assertEqual(self.update(3.2),first)

    def test_track_memory_keeps_last_description(self):
        first=self.update(0)
        for t in (4,5):
            self.assertNotIn('Waiting',self.update(t,[],fast_scene_track_memory=[dict(status='temporarily_missing')]))

    def test_predicted_transition_is_immediate(self):
        self.update(0)
        p=dict(track_id=1,class_name='person',observation_state='predicted',is_predicted=True,
               predicted_bbox=[140,140,160,160],prediction_valid_until=102)
        text=self.update(.1,[],predicted_tracks=[p])
        self.assertIn('previously observed person',text)
        self.assertNotIn('One person is visible',text)

    def test_high_risk_immediate_both_modes(self):
        for mode in ('access','adas'):
            self.fast=FastSceneState(StructuredSceneNarrator(mode=mode),clock=lambda:self.now,wall_clock=lambda:100+self.now)
            self.update(0)
            self.assertTrue(self.update(.1,risk='HIGH').startswith('High risk'))

    def test_new_moving_actor_and_emergence_immediate(self):
        self.update(0)
        text=self.update(.1,stable_motion_evidence=[dict(track_id=1,motion_state='moving_right')])
        self.assertIn('moving right',text)
        text=self.update(.2,predicted_event='actor_may_emerge')
        self.assertIn('An actor may emerge',text)

    def test_duplicates_do_not_count_as_two_observations(self):
        first=self.update(0)
        self.assertEqual(self.update(4,[item(x=20)]),first)
        self.assertEqual(self.update(4,[item(x=20)]),first)
        self.assertIn('left',self.update(4.1,[item(x=20)]))
