"""Deterministic stage 3 checks; no camera or model initialization."""
import unittest
from copy import deepcopy
from fifth_layer.prediction_evaluation import (
    PredictionEvaluationMemory, compare_prediction, evaluation_overlay)
from fifth_layer.world_state import WorldState
from fifth_layer.engine import FifthLayerEngine


def detection(x=100, track=1, state='observed'):
    return dict(track_id=track, class_name='person', observation_state=state,
                box_xyxy=[x-10, 90, x+10, 110])


def register(memory, timestamp=0, source='temporal_live', size=(600, 800)):
    state = WorldState(timestamp, dict(image_width=size[0], image_height=size[1],
                                      detections=[detection()]))
    return memory.record_state(state, dict(track_id=1, motion_state='stationary',
        trajectory=[dict(center=[100, 100], horizon_seconds=1)]), source)[0]


class EvaluationTests(unittest.TestCase):
    def evaluate(self, x, **kwargs):
        m = PredictionEvaluationMemory()
        p = register(m, **kwargs)
        m.observe(1, [detection(x)], *p['image_size'])
        return m.advance(1.25)[0]

    def test_status_thresholds(self):
        for x, status in [(105, 'correct'), (180, 'partially_correct'), (300, 'incorrect')]:
            with self.subTest(x=x):
                self.assertEqual(self.evaluate(x)['status'], status)

    def test_only_same_track_real_observation(self):
        for d in [detection(track=2), detection(state='predicted')]:
            m = PredictionEvaluationMemory()
            register(m)
            m.observe(1, [d], 600, 800)
            self.assertEqual(m.advance(1.25)[0]['status'], 'unevaluable')

    def test_target_grace_nearest_and_once(self):
        m = PredictionEvaluationMemory()
        register(m)
        self.assertEqual(m.observe(.9, [detection(105)], 600, 800), [])
        self.assertEqual(m.advance(1), [])
        self.assertEqual(m.observe(1.02, [detection(102)], 600, 800), [])
        self.assertEqual(m.advance(1.24), [])
        result = m.advance(1.25)[0]
        self.assertEqual(result['center_error_pixels'], 2)
        self.assertEqual(result['observation']['observed_timestamp'], 1.02)
        self.assertEqual(m.advance(2), [])

    def test_missing_and_expired(self):
        m = PredictionEvaluationMemory(ttl=2)
        register(m)
        self.assertEqual(m.advance(1.25)[0]['status'], 'unevaluable')
        register(m, timestamp=2)
        self.assertEqual(m.advance(6)[0]['status'], 'expired')

    def test_normalization(self):
        self.assertEqual(self.evaluate(180)['normalized_center_error'], .08)
        self.assertEqual(self.evaluate(180, size=(1200, 1600))['normalized_center_error'], .04)

    def test_stationary_and_direction(self):
        self.assertTrue(self.evaluate(105)['direction_match'])
        self.assertFalse(self.evaluate(180)['direction_match'])
        m = PredictionEvaluationMemory()
        p = register(m)
        p['predicted_center'] = [200, 100]
        m.observe(1, [detection(190)], 600, 800)
        observation = m.observations[-1]
        self.assertTrue(compare_prediction(p, observation, 1.25)['direction_match'])
        observation['observed_center'] = [0, 100]
        self.assertFalse(compare_prediction(p, observation, 1.25)['direction_match'])

    def test_delayed_deep_preserves_snapshot_and_sources(self):
        m = PredictionEvaluationMemory()
        register(m)
        m.observe(1, [detection()], 600, 800)
        m.advance(2)
        p = register(m, source='temporal_deep')
        self.assertEqual(p['target_timestamp'], 1)
        m.advance(2)
        for source in ('temporal_live', 'temporal_deep'):
            results = m.results_for_source(source)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]['status'], 'correct')
            self.assertEqual(results[0]['observation']['observed_timestamp'], 1)

    def test_limits_ttl_and_no_mutation(self):
        m = PredictionEvaluationMemory(capacity=2, ttl=3)
        for timestamp in range(5):
            register(m, timestamp=timestamp)
            m.observe(timestamp+1, [detection()], 600, 800)
            m.advance(timestamp+1.25)
        self.assertLessEqual(len(m.history), 2)
        self.assertLessEqual(len(m.observations), 2)
        self.assertLessEqual(len(m.pending), 2)
        latest = m.latest(5.25)
        latest['status'] = 'changed'
        self.assertNotEqual(m.history[-1]['status'], 'changed')
        m.advance(20)
        self.assertFalse(m.history)
        self.assertFalse(m.observations)

    def test_engine_legacy_and_optional_comparison(self):
        engine = FifthLayerEngine()
        state = WorldState(data={'value': 1})
        engine.step(state)
        self.assertIsInstance(engine.compare(state).details, dict)
        m = PredictionEvaluationMemory()
        p = register(m)
        self.assertEqual(engine.compare(WorldState(2), prediction=p).details['status'], 'unevaluable')
        with self.assertRaises(ValueError):
            compare_prediction(p, None, .9)

    def test_overlay(self):
        self.assertEqual(evaluation_overlay(self.evaluate(105)), 'EVAL: correct | error 5.0px')
        self.assertEqual(evaluation_overlay(dict(status='unevaluable', center_error_pixels=None)), 'EVAL: unevaluable')
        self.assertEqual(evaluation_overlay(None), '')

    def test_live_overlay_renders_latest_result(self):
        import numpy as np
        from unittest.mock import Mock
        from test_analysis_connections import live_functions
        env = live_functions('draw_overlay')
        cv = Mock()
        memory = PredictionEvaluationMemory()
        register(memory)
        memory.observe(1, [detection(105)], 600, 800)
        env.update(cv2=cv, latest_description='person', latest_motion_summary='stationary',
                   prediction_evaluations=memory, time=Mock(time=lambda: 1.25),
                   get_display_reasoning=lambda: {}, draw_tracked_detections=lambda f, d: f)
        env['draw_overlay'](np.zeros((480, 640, 3)), [])
        labels = [call.args[1] for call in cv.putText.call_args_list]
        self.assertIn('EVAL: correct | error 5.0px', labels)

    def test_occlusion_source(self):
        m = PredictionEvaluationMemory()
        state = WorldState(1, dict(image_width=600, image_height=800, predicted_tracks=[dict(
            track_id=1, class_name='person', predicted_center=[100, 100], predicted_bbox=[90,90,110,110],
            origin_center=[100,100], origin_bbox=[90,90,110,110], position_uncertainty=4)]))
        before = deepcopy(state)
        p = m.record_state(state, {}, 'occlusion_track')[0]
        m.observe(1.1, [detection()], 600, 800)
        self.assertEqual(m.advance(1.25)[0]['status'], 'correct')
        self.assertEqual(p['position_uncertainty_at_prediction'], 4)
        self.assertEqual(state, before)


if __name__ == '__main__':
    unittest.main()
