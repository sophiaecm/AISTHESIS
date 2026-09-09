import unittest
import threading
import tempfile
import os
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from fifth_layer.perception.deep_analysis import DeepAnalysisState
from fifth_layer.perception.analysis_snapshot import AnalysisSnapshot
from fifth_layer.world_state import WorldState
from test_analysis_connections import live_functions


class SceneLifetimeTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.
        self.logger = Mock()
        self.state = DeepAnalysisState(self.logger, lambda: self.now, lambda: 100+self.now)
        self.job = self.state.begin(SimpleNamespace(timestamp=100))

    def test_slow_scene_accepted_temporal_rejected(self):
        self.now = 21
        self.assertTrue(self.state.apply_scene(self.job, 'person at desk'))
        publish = Mock()
        self.assertFalse(self.state.complete(self.job, publish))
        publish.assert_not_called()
        self.assertEqual(self.state.scene_description(), 'person at desk')
        self.assertEqual(self.job['status'], 'timed_out')

    def test_temporal_ttl_does_not_clear_scene(self):
        self.now = 6
        self.state.apply_scene(self.job, 'desk')
        self.state.complete(self.job, lambda: dict(source='temporal_deep'))
        self.state.finish(self.job)
        self.now = 11
        self.assertIsNone(self.state.current())
        self.assertEqual(self.state.scene_description(), 'desk')
        self.now = 31
        self.assertEqual(self.state.scene_description(), 'Analyzing scene...')

    def test_stale_temporal_snapshot_still_valid_for_scene(self):
        self.job['initial_age'] = 10
        self.now = 6
        self.assertTrue(self.state.apply_scene(self.job, 'scene with older snapshot'))
        self.assertFalse(self.state.complete(self.job, Mock()))
        self.assertEqual(self.job['status'], 'stale')
        self.assertEqual(self.state.scene_description(), 'scene with older snapshot')

    def test_stale_scene_does_not_replace_valid_scene(self):
        self.state.apply_scene(self.job, 'valid')
        self.state.finish(self.job)
        self.now = 1
        old = self.state.begin(SimpleNamespace(timestamp=0))
        self.assertFalse(self.state.apply_scene(old, 'obsolete'))
        self.assertEqual(self.state.scene_description(), 'valid')

    def test_session_close_rejects_scene(self):
        self.state.invalidate()
        self.assertFalse(self.state.apply_scene(self.job, 'old session'))

    def environment(self, fail=False):
        env = live_functions('analyze_existing_yolo_result')
        def perceive(_):
            self.now = 21
            if fail:
                raise RuntimeError('inference test failure')
            return WorldState(100, dict(scene_description='measured scene'))
        env.update(deep_analysis_state=self.state, load_smolvlm=lambda: None,
            SMOLVLM_MAX_DIMENSION=448, tempfile=tempfile, os=os, cv2=Mock(),
            smolvlm_lock=threading.Lock(), smolvlm_model=Mock(perceive=perceive),
            perception_fusion=Mock(fuse=lambda a,b: b), orchestrator=Mock(analyze=lambda s: {}),
            prediction_evaluations=Mock(), update_stable_reasoning=Mock())
        return env

    def test_real_analysis_flow_scene_survives_timeout(self):
        env = self.environment()
        snapshot = AnalysisSnapshot.capture(np.zeros((2,2,3), dtype=np.uint8), WorldState(100, {}), [])
        env['analyze_existing_yolo_result'](None,None,snapshot=snapshot,analysis_job=self.job)
        self.assertEqual(self.state.scene_description(), 'measured scene')
        env['prediction_evaluations'].record_state.assert_not_called()
        events = [c.args[0] for c in self.logger.call_args_list]
        self.assertIn('SCENE_INFERENCE_COMPLETED', events)
        self.assertIn('SCENE_DESCRIPTION_APPLIED', events)
        self.assertIn('TEMPORAL_DEEP_TIMED_OUT', events)

    def test_exception_traceback_visible_and_no_ui_exception(self):
        env = self.environment(fail=True)
        snapshot = AnalysisSnapshot.capture(np.zeros((2,2,3), dtype=np.uint8), WorldState(100, {}), [])
        output = StringIO()
        with redirect_stdout(output):
            env['analyze_existing_yolo_result'](None,None,snapshot=snapshot,analysis_job=self.job)
        self.assertIn('SCENE_ANALYSIS_ERROR', output.getvalue())
        self.assertIn('Traceback', output.getvalue())
        self.assertIn('inference test failure', output.getvalue())
        self.assertEqual(self.state.scene_description(), 'Analyzing scene...')

    def test_overlay_does_not_show_expired_global_description(self):
        env = live_functions('draw_overlay')
        cv = Mock()
        env.update(cv2=cv, deep_analysis_state=self.state, latest_description='obsolete global',
            latest_motion_summary='stationary', get_display_reasoning=lambda: {},
            draw_tracked_detections=lambda f,d:f)
        env['draw_overlay'](np.zeros((480,640,3)), [])
        text = ' '.join(c.args[1] for c in cv.putText.call_args_list)
        self.assertIn('Analyzing scene...', text)
        self.assertNotIn('obsolete global', text)
