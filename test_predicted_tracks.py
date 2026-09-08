import unittest
from fifth_layer.perception.tracking import TrackMemory
from fifth_layer.perception.perception_fusion import PerceptionFusion
from fifth_layer.perception.temporal import extract_motion_evidence
from fifth_layer.reasoners.occlusion import OcclusionReasoner
from fifth_layer.reasoners.temporal_prediction import TemporalPredictionReasoner
from fifth_layer.world_state import WorldState


def detection(x, name='person'):
    return dict(class_name=name, confidence=.9, box_xyxy=[x, 100, x+40, 180])


class PredictedTrackTests(unittest.TestCase):
    def setUp(self):
        self.tracker = TrackMemory(observation_expiration=True, predict_missing_tracks=True)

    def update(self, detections, t):
        return self.tracker.update(detections, t, 640, 480)

    def moving(self):
        self.update([detection(100)], 0)
        return self.update([detection(110)], .1)[0]['track_id']

    def test_movement_and_fields(self):
        identity = self.moving()
        self.assertEqual(self.update([], .3), [])
        p = self.tracker.predicted_tracks[0]
        self.assertEqual(p['track_id'], identity)
        self.assertAlmostEqual(p['predicted_center'][0], 150)
        self.assertAlmostEqual(p['predicted_bbox'][0], 130)
        self.assertNotIn('confidence', p)
        self.assertTrue(p['is_predicted'])
        self.assertEqual(self.tracker.tracks[identity]['bbox'][0], 110)

    def test_stationary_and_unreliable(self):
        self.update([detection(100)], 0)
        self.update([], .2)
        p = self.tracker.predicted_tracks[0]
        self.assertEqual(p['predicted_bbox'][0], 100)
        self.assertFalse(p['velocity_reliable'])

    def test_stationary_measured_velocity(self):
        self.update([detection(100)], 0)
        self.update([detection(100)], .1)
        self.update([], .3)
        self.assertEqual(self.tracker.predicted_tracks[0]['predicted_bbox'][0], 100)

    def test_clamping(self):
        self.update([detection(570)], 0)
        self.update([detection(590)], .1)
        self.update([], .5)
        box = self.tracker.predicted_tracks[0]['predicted_bbox']
        self.assertGreaterEqual(min(box), 0)
        self.assertLessEqual(box[2], 640)
        self.assertLessEqual(box[3], 480)
        self.assertEqual(box[2]-box[0], 40)

    def test_uncertainty_and_no_skipped_update(self):
        self.moving()
        self.update([], .2)
        a = self.tracker.predicted_tracks[0]['position_uncertainty']
        age = self.tracker.tracks[0]['missed_frames']
        # UI reads do not perform an extrapolation step.
        for _ in range(20):
            self.assertEqual(self.tracker.tracks[0]['missed_frames'], age)
        self.update([], .4)
        self.assertGreater(self.tracker.predicted_tracks[0]['position_uncertainty'], a)

    def test_reappearance_current_error_only(self):
        identity = self.moving()
        self.update([], .2)
        observed = self.update([detection(133)], .3)[0]
        self.assertEqual(observed['track_id'], identity)
        self.assertEqual(observed['observation_state'], 'observed')
        self.assertFalse(observed['is_predicted'])
        self.assertAlmostEqual(observed['reappearance_error_pixels'], 3)
        self.assertEqual(self.tracker.predicted_tracks, [])
        next_observed = self.update([detection(134)], .4)[0]
        self.assertNotIn('reappearance_error_pixels', next_observed)

    def test_deforming_box_freezes_prediction(self):
        self.update([detection(100)], 0)
        wide = detection(100)
        wide['box_xyxy'][2] = 220
        self.update([wide], .1)
        self.update([], .3)
        p = self.tracker.predicted_tracks[0]
        self.assertFalse(p['velocity_reliable'])
        self.assertEqual(p['predicted_bbox'], [100, 100, 220, 180])

    def test_predicted_overlay_label_and_expiry(self):
        from unittest.mock import Mock
        import numpy as np
        from test_analysis_connections import live_functions
        env = live_functions('draw_overlay')
        self.moving()
        self.update([], .3)
        cv = Mock()
        clock = Mock(time=lambda: .3)
        env.update(cv2=cv, time=clock, latest_description='', latest_motion_summary='',
                   latest_stable_motion_evidence=[], prediction_feedback=Mock(history=[]),
                   get_display_reasoning=lambda: {}, draw_tracked_detections=lambda frame, _: frame)
        env['draw_overlay'](np.zeros((480, 640, 3)), [], self.tracker.predicted_tracks)
        labels = [c.args[1] for c in cv.putText.call_args_list]
        self.assertIn('PREDICTED person #0', labels)
        cv.reset_mock()
        clock.time = lambda: 2
        env['draw_overlay'](np.zeros((480, 640, 3)), [], self.tracker.predicted_tracks)
        self.assertFalse(any(c.args[1].startswith('PREDICTED person') for c in cv.putText.call_args_list))

    def test_normal_timeout(self):
        self.moving()
        self.update([], 1.61)
        self.assertEqual(self.tracker.predicted_tracks, [])

    def test_possible_occlusion_extended_timeout(self):
        self.moving()
        self.update([detection(140, 'chair')], .4)
        p = self.tracker.predicted_tracks[0]
        self.assertEqual(p['evidence'], 'possible_occlusion')
        self.assertEqual(p['prediction_valid_until'], 2.6)
        self.update([detection(140, 'chair')], 2)
        self.assertTrue(self.tracker.predicted_tracks)
        self.update([detection(140, 'chair')], 2.61)
        self.assertFalse(self.tracker.predicted_tracks)

    def test_two_predictions(self):
        self.update([detection(100), detection(400)], 0)
        self.update([detection(110), detection(390)], .1)
        self.update([], .2)
        positions = {p['track_id']: p['predicted_center'][0] for p in self.tracker.predicted_tracks}
        self.assertEqual(positions, {0: 140, 1: 400})

    def test_separate_fusion_and_reasoner_context(self):
        self.moving()
        accepted = self.update([], .2)
        state = WorldState(.2, dict(detections=accepted, accepted_detections=accepted,
            detection_count=0, image_width=640, image_height=480,
            predicted_tracks=self.tracker.predicted_tracks))
        fused = PerceptionFusion().fuse(state, WorldState(data={'scene_description': ''}))
        self.assertEqual(fused.data['detection_count'], 0)
        self.assertEqual(fused.data['detections'], [])
        self.assertEqual(fused.data['semantic_evidence'], PerceptionFusion().fuse(
            WorldState(data={'detections': []}), WorldState(data={'scene_description': ''})).data['semantic_evidence'])
        for reasoner in (OcclusionReasoner(), TemporalPredictionReasoner()):
            result = reasoner.infer_expected_consequences(state).predictions
            self.assertTrue(result['predicted_track_context'])
            self.assertNotIn('predicted_track_context', reasoner.infer_expected_consequences(WorldState()).predictions)
        self.assertEqual(extract_motion_evidence(self.tracker.predicted_tracks,
                         self.tracker.predicted_tracks, 640, 480, .1), [])
        before = self.tracker.next_id
        self.assertEqual(self.update(self.tracker.predicted_tracks, .3), [])
        self.assertEqual(self.tracker.next_id, before)


if __name__ == '__main__':
    unittest.main()
