"""Live integration regressions without camera or model downloads."""
import threading
import unittest
from unittest.mock import Mock
import numpy as np
from fifth_layer.perception.live_inference import LiveInferenceWorker
from fifth_layer.perception.temporal import extract_motion_evidence
from fifth_layer.world_state import WorldState
from test_analysis_connections import live_functions


class LiveIntegrationTests(unittest.TestCase):
    def test_evaluation_reaches_overlay_before_camera_loop_closes(self):
        from fifth_layer.perception.tracking import TrackMemory
        from fifth_layer.prediction_evaluation import PredictionEvaluationMemory

        for x, status in [(105, 'correct'), (180, 'partially_correct'), (300, 'incorrect')]:
            with self.subTest(status=status):
                env = live_functions('run_live_camera', 'track_observation',
                                     'filter_live_tracking_detections', 'draw_overlay')
                memory = PredictionEvaluationMemory()
                tracker = TrackMemory(max_distance=1, predict_missing_tracks=True)
                frame = np.zeros((800, 600, 3))
                states = [WorldState(t, dict(image_width=600, image_height=800,
                    detections=[dict(class_name='person', confidence=.9,
                                     box_xyxy=[cx-10, 90, cx+10, 110])]))
                    for t, cx in [(0, 100), (1.25, x)]]
                clock = [0.0]
                worker, cv = Mock(), Mock()
                forecast_ids, rendered = [], []
                worker.poll.side_effect = [(frame, 0, object(), None),
                                           (frame, 1.25, object(), None), None]
                def read():
                    clock[0] = [0, 1.25, 1.30][cv.VideoCapture.return_value.read.call_count-1]
                    return True, frame
                cv.VideoCapture.return_value.read.side_effect = read

                def forecast(state):
                    if state.timestamp == 0:
                        records = memory.record_state(state, dict(
                            track_id=state.data['detections'][0]['track_id'],
                            motion_state='stationary', trajectory=[dict(
                                center=[100, 100], horizon_seconds=1)]), 'temporal_live')
                        forecast_ids.append(records[0]['prediction_id'])
                        self.assertEqual(len(memory.pending), 1)
                    else:
                        # observe() must finalize before the overlay or shutdown.
                        self.assertFalse(memory.pending)
                        self.assertEqual(state.data['prediction_evaluations'][0]['status'], status)
                        worker.close.assert_not_called()

                def wait_key(_):
                    iteration = cv.waitKey.call_count
                    worker.close.assert_not_called()
                    cv.VideoCapture.return_value.release.assert_not_called()
                    labels = [call.args[1] for call in cv.putText.call_args_list
                              if call.args[1].startswith('EVAL:')]
                    if iteration == 1:
                        self.assertFalse(labels)
                        self.assertEqual(len(memory.pending), 1)
                    else:
                        self.assertEqual(len(memory.history), 1)
                        self.assertEqual(memory.history[0]['prediction_id'], forecast_ids[0])
                        self.assertEqual(memory.history[0]['observation']['observation_state'], 'observed')
                        self.assertEqual(labels[-1], f'EVAL: {status} | error {x-100:.1f}px')
                        rendered.append(labels[-1])
                    return ord('q') if iteration == 3 else -1

                cv.waitKey.side_effect = wait_key
                env.update(cv2=cv, time=Mock(time=lambda: clock[0]), CAMERA_INDEX=0,
                    reset_motion_smoothing=lambda: (tracker.reset(), memory.reset()),
                    object_tracker=tracker, prediction_evaluations=memory,
                    LIVE_MIN_TRACK_CONFIDENCE=.5,
                    LiveInferenceWorker=Mock(return_value=worker), infer_yolo=Mock(),
                    build_yolo_world_state=Mock(side_effect=states),
                    calculate_temporal_motion=Mock(), update_live_temporal_prediction=forecast,
                    analysis_running=True, LIVE_YOLO_EVERY_N_FRAMES=2,
                    LIVE_YOLO_EVERY_N_FRAMES_DURING_VLM=4,
                    get_display_reasoning=lambda: {}, draw_tracked_detections=lambda f, d: f)
                env['run_live_camera']()
                self.assertEqual(len(rendered), 2)
                worker.close.assert_called_once()
                cv.VideoCapture.return_value.release.assert_called_once()

    def test_worker_does_not_wait_for_slow_model_or_close(self):
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()
        def infer(frame):
            entered.set()
            release.wait(2)
            finished.set()
            return 'result'
        worker = LiveInferenceWorker(infer)
        try:
            self.assertTrue(worker.submit(np.zeros((2, 2, 3)), 1))
            self.assertTrue(entered.wait(1))
            self.assertIsNone(worker.poll())
            self.assertFalse(worker.submit(np.zeros((2, 2, 3)), 2))
            worker.close()
            self.assertFalse(finished.is_set())
        finally:
            release.set()

    def test_worker_preserves_frame_timestamp_and_reports_errors(self):
        worker = LiveInferenceWorker(lambda frame: 1 / 0)
        try:
            frame = np.zeros((2, 2, 3))
            worker.submit(frame, 12)
            frame[:] = 1
            # Wait only in the test; production UI uses nonblocking poll.
            observed, timestamp, value, error = worker.results.get(timeout=1)
            self.assertEqual(observed.sum(), 0)
            self.assertEqual(timestamp, 12)
            self.assertIsNone(value)
            self.assertIsInstance(error, ZeroDivisionError)
        finally:
            worker.close()

    def test_overlay_draws_ids_after_panel(self):
        env = live_functions('draw_overlay')
        cv = Mock()
        order = []
        cv.putText.side_effect = lambda *args: order.append('panel')
        detections = [{'track_id': 3}]
        def draw(frame, items):
            self.assertIs(items, detections)
            order.append('boxes')
            return frame
        env.update(cv2=cv, latest_description='person', latest_motion_summary='stationary',
                   latest_stable_motion_evidence=[], prediction_feedback=Mock(history=[]),
                   get_display_reasoning=lambda: {}, draw_tracked_detections=draw)
        env['draw_overlay'](np.zeros((480, 640, 3)), detections)
        self.assertEqual(order[-1], 'boxes')

    def test_live_loop_pumps_events_and_reuses_world_detections(self):
        env = live_functions('run_live_camera')
        frame = np.zeros((480, 640, 3))
        state = WorldState(10, dict(image_width=640, image_height=480, detections=[]))
        tracked = [{'track_id': 3, 'class_name': 'person', 'confidence': .87}]
        def track(world):
            world.data['detections'] = tracked
            return tracked
        worker = Mock()
        worker.poll.side_effect = [None, (frame, 10, object(), None), None, None]
        cv = Mock()
        cv.VideoCapture.return_value.read.return_value = (True, frame)
        cv.waitKey.side_effect = [-1, -1, -1, ord('q')]
        overlay = Mock(return_value=frame)
        env.update(cv2=cv, time=Mock(time=lambda: 10), CAMERA_INDEX=0,
                   reset_motion_smoothing=Mock(), LiveInferenceWorker=Mock(return_value=worker),
                   infer_yolo=Mock(), build_yolo_world_state=Mock(return_value=state),
                   track_observation=track, calculate_temporal_motion=Mock(),
                   object_tracker=Mock(tracks={}),
                   update_live_temporal_prediction=Mock(), analysis_running=True,
                   LIVE_YOLO_EVERY_N_FRAMES=2, LIVE_YOLO_EVERY_N_FRAMES_DURING_VLM=4,
                   draw_overlay=overlay)
        env['run_live_camera']()
        self.assertEqual(cv.waitKey.call_count, 4)
        self.assertEqual(env['track_observation'], track)
        for call in overlay.call_args_list[1:]:
            self.assertIs(call.args[1], state.data['detections'])
        worker.close.assert_called_once()
        cv.VideoCapture.return_value.release.assert_called_once()
        env['infer_yolo'].assert_not_called()

    def test_live_jitter_deadband_preserves_legacy_and_real_motion(self):
        previous = [dict(class_name='person', box_xyxy=[100, 100, 300, 400])]
        def current(dx):
            return [dict(class_name='person', box_xyxy=[100+dx, 100, 300+dx, 400])]
        legacy = extract_motion_evidence(previous, current(6), 640, 480, .05)
        self.assertEqual(legacy[0]['motion_state'], 'moving_right')
        jitter = extract_motion_evidence(previous, current(6), 640, 480, .05,
                                         minimum_motion_pixels=8, minimum_normalized_motion=.01)
        self.assertEqual(jitter[0]['motion_state'], 'stationary')
        moving = extract_motion_evidence(previous, current(20), 640, 480, .05,
                                         minimum_motion_pixels=8, minimum_normalized_motion=.01)
        self.assertEqual(moving[0]['motion_state'], 'moving_right')


if __name__ == '__main__':
    unittest.main()
