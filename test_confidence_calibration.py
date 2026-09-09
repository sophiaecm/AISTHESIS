"""Stage 4 session calibration regressions without model initialization."""
import unittest
from copy import deepcopy
from unittest.mock import Mock
from fifth_layer.confidence_calibration import ConfidenceCalibrationMemory, calibration_overlay
from fifth_layer.prediction_evaluation import PredictionEvaluationMemory
from fifth_layer.world_state import WorldState
from test_prediction_evaluation import register, detection


def outcome(i, status='correct', source='temporal_live', kind='trajectory_position', label='person'):
    return dict(prediction_id=str(i), evaluated_timestamp=float(i), status=status,
                prediction=dict(source=source, prediction_type=kind, class_name=label))


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.memory = ConfidenceCalibrationMemory()

    def confidence(self, source='temporal_live', kind='trajectory_position', label='person', raw=.72, timestamp=20):
        return self.memory.calibrate(raw, source, kind, label, timestamp)

    def test_prior_and_first_samples_leave_raw_unchanged(self):
        for i in range(4):
            self.memory.update(outcome(i))
            c = self.confidence(timestamp=i)
            self.assertEqual(c['calibrated_confidence'], .72)
            self.assertEqual(c['calibration_samples'], i+1)
        self.assertEqual(c['calibration_bucket'], ['global_prior'])

    def test_five_successes_raise_confidence_with_beta_smoothing(self):
        for i in range(5):
            self.memory.update(outcome(i))
        c = self.confidence()
        self.assertAlmostEqual(c['calibration_reliability'], 7/9)
        self.assertAlmostEqual(c['calibrated_confidence'], .8)
        self.assertEqual(c['raw_confidence'], .72)

    def test_single_failure_is_limited_and_repeated_failures_reduce(self):
        for i in range(5):
            self.memory.update(outcome(i))
        before = self.confidence(timestamp=4)['calibrated_confidence']
        self.memory.update(outcome(5, 'incorrect'))
        after = self.confidence(timestamp=5)['calibrated_confidence']
        self.assertLess(before-after, .03)
        for i in range(6, 20):
            self.memory.update(outcome(i, 'incorrect'))
            current = self.confidence(timestamp=i)['calibrated_confidence']
            self.assertLess(current, after)
            after = current
        self.assertLess(after, .72)

    def test_partial_weight_and_bucket_statistics(self):
        self.memory.update(outcome(0, 'partially_correct'))
        b = self.memory.statistics()[0]
        self.assertEqual(b['weighted_success'], .5)
        self.assertEqual(b['partial_count'], 1)
        self.assertEqual(b['total_evaluable'], 1)
        self.assertEqual(b['reliability'], .5)

    def test_missing_expired_and_duplicates_do_not_update(self):
        r = outcome(1)
        self.assertTrue(self.memory.update(r))
        before = self.memory.statistics()
        self.assertFalse(self.memory.update(r))
        for status in ('unevaluable', 'expired'):
            self.assertFalse(self.memory.update(outcome(2, status)))
        self.assertEqual(before, self.memory.statistics())

    def test_source_isolation(self):
        for i in range(10):
            self.memory.update(outcome(i, 'incorrect'))
        for source in ('temporal_deep', 'occlusion_track'):
            c = self.confidence(source=source)
            self.assertEqual(c['calibrated_confidence'], .72)
            self.assertEqual(c['calibration_samples'], 0)

    def test_hierarchical_fallback(self):
        for i in range(5):
            self.memory.update(outcome(i, label=str(i)))
        self.assertEqual(self.confidence()['calibration_bucket'], ['temporal_live', 'trajectory_position'])
        self.assertEqual(self.confidence(kind='other')['calibration_bucket'], ['temporal_live'])
        for i in range(5, 10):
            self.memory.update(outcome(i))
        self.assertEqual(self.confidence()['calibration_bucket'], ['temporal_live', 'trajectory_position', 'person'])

    def test_bounds_and_relative_effect(self):
        for status in ('correct', 'incorrect'):
            self.memory.reset()
            for i in range(100):
                self.memory.update(outcome(i, status))
            for raw in (0, .1, .72, .99, 1):
                value = self.confidence(raw=raw, timestamp=100)['calibrated_confidence']
                self.assertGreaterEqual(value, 0)
                self.assertLessEqual(value, 1)
                self.assertGreaterEqual(value, raw*.8-1e-12)
                self.assertLessEqual(value, raw*1.2+1e-12)

    def test_memory_limits_ttl_and_replay_after_eviction(self):
        self.memory = ConfidenceCalibrationMemory(max_buckets=3, max_ids=2, ttl=10)
        for i in range(8):
            self.memory.update(outcome(i, kind=str(i)))
        self.assertLessEqual(len(self.memory.buckets), 3)
        self.assertLessEqual(len(self.memory.processed), 2)
        self.assertFalse(self.memory.update(outcome(0)))
        self.confidence(timestamp=30)
        self.assertFalse(self.memory.buckets)
        self.assertFalse(self.memory.processed)
        self.assertFalse(self.memory.update(outcome(7)))

    def test_session_reset(self):
        for i in range(5):
            self.memory.update(outcome(i))
        self.memory.reset()
        self.assertEqual(self.confidence()['calibration_samples'], 0)
        self.assertEqual(self.confidence()['calibration_reliability'], .5)

    def test_pending_does_not_learn_and_past_forecasts_are_immutable(self):
        m = PredictionEvaluationMemory()
        p = register(m)
        original = deepcopy(p)
        self.assertFalse(m.calibration.statistics())
        m.observe(1, [detection()], 600, 800)
        self.assertFalse(m.calibration.statistics())
        m.advance(1.25)
        self.assertEqual(m.history[0]['prediction'], original)
        self.assertEqual(m.calibration.statistics()[0]['total_evaluable'], 1)
        for i in range(1, 5):
            register(m, timestamp=i*2)
            m.observe(i*2+1, [detection()], 600, 800)
            m.advance(i*2+1.25)
        future = register(m, timestamp=10)
        self.assertGreater(future['calibrated_confidence'], future['raw_confidence'])
        self.assertEqual(p, original)

    def test_overlay_uses_issuance_values(self):
        p = self.confidence()
        self.assertEqual(calibration_overlay(dict(prediction=p)), 'CAL: collecting evidence | n=0/5')
        for i in range(5):
            self.memory.update(outcome(i+21))
        p = self.confidence(timestamp=26)
        self.assertEqual(calibration_overlay(dict(prediction=p)), 'CAL: raw 0.72 -> calibrated 0.80 | n=5')

    def test_detector_and_tracker_unchanged(self):
        from fifth_layer.perception.tracking import TrackMemory
        tracker = TrackMemory(predict_missing_tracks=True)
        d = detection()
        d['confidence'] = .91
        first = tracker.update([d], 0, 600, 800)
        before = deepcopy(tracker.tracks)
        state = WorldState(0, dict(image_width=600, image_height=800, detections=first))
        m = PredictionEvaluationMemory()
        m.record_state(state, dict(track_id=first[0]['track_id'], raw_confidence=.72,
                       trajectory=[dict(center=[100, 100], horizon_seconds=1)]), 'temporal_live')
        self.assertEqual(tracker.tracks, before)
        self.assertEqual(first[0]['confidence'], .91)
        second = tracker.update([d], .1, 600, 800)
        self.assertEqual(first[0]['track_id'], second[0]['track_id'])

    def test_legacy_world_state_and_fusion_unchanged(self):
        from fifth_layer.reasoners.sensor_fusion import SensorFusionReasoner
        state = WorldState(data=dict(vision_hidden_actor_possible=True, vision_confidence=.72))
        before = deepcopy(state)
        reasoner = SensorFusionReasoner()
        first = reasoner.infer_expected_consequences(state)
        state.data['evaluation_predictions'] = [self.confidence()]
        second = reasoner.infer_expected_consequences(state)
        self.assertEqual(first, second)
        self.assertEqual(state.data['vision_confidence'], before.data['vision_confidence'])

    def test_calibration_event_only_on_new_evaluable_result(self):
        logger = Mock()
        m = ConfidenceCalibrationMemory(event_logger=logger)
        m.update(outcome(1, 'unevaluable'))
        logger.assert_not_called()
        m.update(outcome(2))
        m.update(outcome(2))
        logger.assert_called_once()
        self.assertEqual(logger.call_args.args[0], 'CONFIDENCE_CALIBRATION')

    def test_actual_overlay_collecting_and_calibrated(self):
        import numpy as np
        from test_analysis_connections import live_functions
        for count in (0, 5):
            with self.subTest(count=count):
                m = PredictionEvaluationMemory()
                for i in range(count):
                    m.calibration.update(outcome(i))
                p = register(m, timestamp=10)
                m.observe(11, [detection()], 600, 800)
                env = live_functions('draw_overlay')
                cv = Mock()
                env.update(cv2=cv, prediction_evaluations=m, time=Mock(time=lambda: 11.25),
                    latest_description='person', latest_motion_summary='stationary',
                    get_display_reasoning=lambda: {}, draw_tracked_detections=lambda f, d: f)
                env['draw_overlay'](np.zeros((800, 600, 3)), [])
                labels = [c.args[1] for c in cv.putText.call_args_list if c.args[1].startswith('CAL:')]
                self.assertEqual(labels, [calibration_overlay(dict(prediction=p))])
                self.assertIn('collecting evidence' if count == 0 else 'raw 0.50', labels[0])

    def test_idle_overlay_tick_cleans_calibration_memory(self):
        m = PredictionEvaluationMemory()
        m.calibration.update(outcome(0))
        m.latest(301)
        self.assertFalse(m.calibration.buckets)
        self.assertFalse(m.calibration.processed)


if __name__ == '__main__':
    unittest.main()
