import unittest
from unittest.mock import Mock
import numpy as np
from fifth_layer.world_state import WorldState
from fifth_layer.perception.tracking import TrackMemory
from fifth_layer.perception.structured_scene_narrator import FastSceneState, StructuredSceneNarrator
from test_analysis_connections import live_functions


def observed(timestamp=100):
    return WorldState(timestamp, dict(image_width=640,image_height=480,
        accepted_detections=[dict(track_id=0,class_name='person',confidence=.92,
            observation_state='observed',box_xyxy=[270,190,370,290])]))


class FastConnectionTests(unittest.TestCase):
    def test_first_accepted_and_debounce(self):
        f=FastSceneState(clock=lambda:0,wall_clock=lambda:100)
        f.changed=0
        f.update(observed(),completed_observation=True)
        self.assertTrue(f.description().startswith('One person is visible near the center of the scene'))

    def test_missing_timestamp_completed_only_and_input_preserved(self):
        f=FastSceneState(wall_clock=lambda:100)
        state=observed(None)
        self.assertIsNone(f.update(state))
        self.assertIsNotNone(f.update(state,completed_observation=True))
        self.assertIsNone(state.timestamp)
        self.assertNotIn('Waiting',f.description())

    def test_empty_start_does_not_block_first_detection(self):
        f=FastSceneState(clock=lambda:0,wall_clock=lambda:100)
        self.assertIsNone(f.update(WorldState(100,dict(accepted_detections=[]))))
        self.assertIn('Waiting',f.description())
        self.assertIsNotNone(f.update(observed()))
        self.assertNotIn('Waiting',f.description())

    def test_silent_age_rejection_is_now_diagnosable(self):
        logger=Mock()
        f=FastSceneState(wall_clock=lambda:105,logger=logger)
        self.assertIsNone(f.update(observed()))
        self.assertEqual(logger.call_args.kwargs['update_reason'],'observation_age_exceeded')
        self.assertEqual(logger.call_args.kwargs['accepted_detections_count'],1)
        self.assertEqual(logger.call_args.kwargs['observation_age_seconds'],5)

    def test_camera_and_video_use_same_tracked_list_before_overlay(self):
        for mode in ('run_live_camera','open_video'):
            with self.subTest(mode=mode):
                env=live_functions(mode,'track_observation','filter_live_tracking_detections','draw_overlay')
                frame=np.zeros((480,640,3))
                state=observed()
                state.data['detections']=state.data.pop('accepted_detections')
                state.data['detections'][0].pop('track_id')
                cv,worker=Mock(),Mock()
                cv.VideoCapture.return_value.read.return_value=(True,frame)
                cv.waitKey.return_value=ord('q')
                worker.poll.return_value=(frame,100,object(),None)
                tracker=TrackMemory(predict_missing_tracks=True)
                narrator=StructuredSceneNarrator(detail_level='detailed')
                real_narrate=narrator.narrate
                seen=[]
                def narrate(ws):
                    self.assertIs(ws.data['accepted_detections'],state.data['detections'])
                    self.assertEqual(ws.data['accepted_detections'][0]['track_id'],0)
                    seen.append(ws.data['accepted_detections'])
                    return real_narrate(ws)
                narrator.narrate=narrate
                def boxes(f,ds):
                    self.assertIs(ds,seen[0])
                    return f
                fast=FastSceneState(narrator=narrator,wall_clock=lambda:100)
                env.update(cv2=cv,time=Mock(time=lambda:100),CAMERA_INDEX=0,
                    reset_motion_smoothing=Mock(),LiveInferenceWorker=Mock(return_value=worker),
                    infer_yolo=Mock(),build_yolo_world_state=Mock(return_value=state),
                    object_tracker=tracker,LIVE_MIN_TRACK_CONFIDENCE=.5,
                    fast_scene_state=fast,calculate_temporal_motion=Mock(),
                    update_live_temporal_prediction=Mock(),analysis_running=True,
                    LIVE_YOLO_EVERY_N_FRAMES=2,LIVE_YOLO_EVERY_N_FRAMES_DURING_VLM=4,
                    get_display_reasoning=lambda:{},draw_tracked_detections=boxes,
                    filedialog=Mock(askopenfilename=lambda **kw:'video.mp4'),
                    run_yolo=lambda f:(f,object()),DESCRIPTION_INTERVAL_SECONDS=999)
                env[mode]()
                self.assertEqual(len(seen),1)
                self.assertIn('One person',fast.description())
                labels=' '.join(c.args[1] for c in cv.putText.call_args_list)
                self.assertIn('One person',labels)
                self.assertNotIn('Waiting for a fresh structured observation',labels)
                self.assertIsNone(env['deep_analysis_state'].scene)
