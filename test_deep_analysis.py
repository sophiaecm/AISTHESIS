"""Stage 5 admission, lifecycle and live selection tests; no models required."""
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from fifth_layer.perception.deep_analysis import DeepAnalysisState
from test_analysis_connections import live_functions


class DeepAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.
        self.wall = 100.
        self.logger = Mock()
        self.state = DeepAnalysisState(self.logger, lambda: self.now, lambda: self.wall)

    def begin(self, timestamp=100):
        return self.state.begin(SimpleNamespace(timestamp=timestamp))

    def publish(self):
        return dict(source='temporal_deep', prediction='moving', track_id=3, prediction_id='p1')

    def test_fresh_metadata_and_events(self):
        job = self.begin()
        self.now = 10
        self.assertTrue(self.state.complete(job, self.publish))
        result = self.state.current()
        self.assertEqual(result['status'], 'fresh')
        self.assertEqual(result['result_age_seconds'], 10)
        self.assertEqual(result['snapshot_timestamp'], 100)
        self.assertEqual(result['snapshot_sequence_id'], 1)
        self.assertEqual(result['prediction_id'], 'p1')
        events = [c.args[0] for c in self.logger.call_args_list]
        self.assertEqual(events, ['TEMPORAL_DEEP_STARTED', 'SCENE_ANALYSIS_STARTED', 'TEMPORAL_DEEP_APPLIED'])

    def test_stale_no_publication(self):
        job = self.begin(timestamp=90)
        self.now = 6
        publish = Mock()
        self.assertFalse(self.state.complete(job, publish))
        publish.assert_not_called()
        self.assertEqual(job['status'], 'stale')

    def test_timeout_no_publication_and_slot_not_released_early(self):
        job = self.begin()
        self.now = 12.1
        self.assertIsNone(self.state.current())
        self.assertEqual(job['status'], 'timed_out')
        publish = Mock()
        self.assertFalse(self.state.complete(job, publish))
        publish.assert_not_called()
        self.assertIsNone(self.begin())
        self.state.finish(job)
        self.assertIsNotNone(self.begin())

    def test_older_sequence_superseded(self):
        old = self.begin()
        self.state.finish(old)
        new = self.begin()
        self.state.complete(new, self.publish)
        publish = Mock()
        self.assertFalse(self.state.complete(old, publish))
        publish.assert_not_called()
        self.assertEqual(old['status'], 'superseded')
        self.assertEqual(self.state.current()['snapshot_sequence_id'], new['sequence'])

    def test_display_ttl_and_live_eval_cal_preserved(self):
        job = self.begin()
        self.state.complete(job, self.publish)
        self.state.finish(job)
        env = live_functions('get_display_reasoning')
        live = dict(source='temporal_live', prediction='live_motion')
        memory = env['prediction_evaluations']
        memory.history.append(dict(status='correct', prediction_id='retained'))
        env.update(deep_analysis_state=self.state, latest_reasoning=live,
            display_reasoning={}, display_reasoning_last_valid_at=0.,
            PREDICTION_DISPLAY_HOLD_SECONDS=100)
        self.assertEqual(env['get_display_reasoning']()['source'], 'temporal_live')
        self.now = 5
        self.assertEqual(env['get_display_reasoning'](), live)
        self.assertIsNone(self.state.current())
        self.assertEqual(memory.history[0]['prediction_id'], 'retained')
        self.assertIs(env['latest_reasoning'], live)

    def test_display_cache_cannot_resurrect_expired_deep(self):
        job = self.begin()
        self.state.complete(job, self.publish)
        self.state.finish(job)
        env = live_functions('get_display_reasoning')
        env.update(deep_analysis_state=self.state,
            latest_reasoning=dict(prediction='indeterminate'), display_reasoning={},
            display_reasoning_last_valid_at=0., PREDICTION_DISPLAY_HOLD_SECONDS=100)
        self.assertEqual(env['get_display_reasoning']()['source'], 'temporal_deep')
        self.now = 5
        self.assertEqual(env['get_display_reasoning']()['prediction'], 'indeterminate')

    def test_clock_jump_does_not_change_elapsed_age(self):
        job = self.begin()
        self.wall = -10000
        self.now = 2
        self.state.complete(job, self.publish)
        self.assertEqual(self.state.current()['result_age_seconds'], 2)
        self.wall = 10000
        self.now = 3
        self.assertEqual(self.state.current()['result_age_seconds'], 3)

    def test_shutdown_bounded_and_late_result_discarded(self):
        job = self.begin()
        release = threading.Event()
        worker = threading.Thread(target=lambda: release.wait(2), daemon=True)
        self.state.worker = worker
        worker.start()
        try:
            self.state.shutdown()
            self.assertTrue(worker.is_alive())
            self.assertEqual(job['status'], 'superseded')
            publish = Mock()
            self.assertFalse(self.state.complete(job, publish))
            publish.assert_not_called()
        finally:
            release.set()
            worker.join(1)

    def test_single_slot_no_queue(self):
        job = self.begin()
        for _ in range(100):
            self.assertIsNone(self.begin())
        self.assertEqual(self.state.sequence, 1)
        self.state.invalidate()
        self.assertIsNone(self.begin())
        self.state.finish(job)
        self.assertIsNotNone(self.begin())

    def test_camera_events_continue_while_deep_worker_is_blocked(self):
        import numpy as np
        release = threading.Event()
        self.begin()
        worker = threading.Thread(target=lambda: release.wait(3), daemon=True)
        self.state.worker = worker
        worker.start()
        env = live_functions('run_live_camera')
        cv, inference = Mock(), Mock()
        cv.VideoCapture.return_value.read.return_value = (True, np.zeros((2, 2, 3)))
        inference.poll.return_value = None
        def key(_):
            self.assertTrue(worker.is_alive())
            return ord('q') if cv.waitKey.call_count == 3 else -1
        cv.waitKey.side_effect = key
        env.update(cv2=cv, CAMERA_INDEX=0, reset_motion_smoothing=Mock(),
            deep_analysis_state=self.state, LiveInferenceWorker=Mock(return_value=inference),
            infer_yolo=Mock(), analysis_running=True, LIVE_YOLO_EVERY_N_FRAMES=2,
            LIVE_YOLO_EVERY_N_FRAMES_DURING_VLM=4, draw_overlay=lambda f, *a: f)
        try:
            env['run_live_camera']()
            self.assertEqual(cv.waitKey.call_count, 3)
            self.assertTrue(worker.is_alive())
            cv.VideoCapture.return_value.release.assert_called_once()
        finally:
            release.set()
            worker.join(1)

    def test_rejected_background_never_records_evaluation_or_ui(self):
        import numpy as np
        import tempfile
        import os
        from fifth_layer.world_state import WorldState
        from fifth_layer.perception.analysis_snapshot import AnalysisSnapshot
        for reason in ('stale', 'timed_out', 'superseded'):
            with self.subTest(reason=reason):
                self.setUp()
                snapshot = AnalysisSnapshot.capture(np.zeros((2, 2, 3), dtype=np.uint8),
                    WorldState(100, {}), [])
                job = self.begin(timestamp=90 if reason == 'stale' else 100)
                env = live_functions('analyze_existing_yolo_result')
                def perceive(_):
                    self.now = 6 if reason == 'stale' else 13 if reason == 'timed_out' else 1
                    if reason == 'superseded':
                        self.state.invalidate()
                    return WorldState(100, {})
                evaluation = Mock()
                update = Mock()
                env.update(deep_analysis_state=self.state, load_smolvlm=lambda: None,
                    SMOLVLM_MAX_DIMENSION=448, tempfile=tempfile, os=os, cv2=Mock(),
                    smolvlm_lock=threading.Lock(), smolvlm_model=Mock(perceive=perceive),
                    perception_fusion=Mock(fuse=lambda a, b: WorldState(100, {})),
                    orchestrator=Mock(analyze=lambda s: {}), prediction_evaluations=evaluation,
                    update_stable_reasoning=update, update_stable_description=update)
                env['analyze_existing_yolo_result'](None, None, snapshot=snapshot, analysis_job=job)
                evaluation.record_state.assert_not_called()
                evaluation.calibration.update.assert_not_called()
                update.assert_not_called()
                self.assertEqual(job['status'], reason)


if __name__ == '__main__':
    unittest.main()
