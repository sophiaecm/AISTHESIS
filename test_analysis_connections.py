"""Regression tests without loading camera, YOLO, or SmolVLM models."""
import ast
import copy
from dataclasses import FrozenInstanceError
from pathlib import Path
import threading
import unittest
from unittest.mock import Mock

import numpy as np

from fifth_layer.perception.analysis_snapshot import AnalysisSnapshot
from fifth_layer.reasoners.orchestrator import AisthesisOrchestrator
from fifth_layer.reasoners.sensor_fusion import SensorFusionReasoner
from fifth_layer.reasoners.temporal_prediction import TemporalPredictionReasoner
from fifth_layer.world_state import WorldState


def live_functions(*names):
    tree = ast.parse(Path("live_app.py").read_text())
    tree.body = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    from fifth_layer.prediction_evaluation import PredictionEvaluationMemory, evaluation_overlay
    import time
    env = {"AnalysisSnapshot": AnalysisSnapshot,
           "prediction_evaluations": PredictionEvaluationMemory(),
           "evaluation_overlay": evaluation_overlay, "time": time}
    exec(compile(tree, "live_app.py", "exec"), env)
    return env


class SnapshotTests(unittest.TestCase):
    def test_immutable_nested_snapshot(self):
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        state = WorldState(12.5, {"detections": [{"box_xyxy": [1, 2, 3, 4]}]})
        motion = [{"current_center": [2, 3]}]
        snapshot = AnalysisSnapshot.capture(frame, state, motion)
        frame[:] = 255
        state.data["detections"][0]["box_xyxy"][0] = 99
        motion[0]["current_center"][0] = 99
        self.assertEqual(snapshot.frame().sum(), 0)
        self.assertEqual(snapshot.world_state().data["detections"][0]["box_xyxy"][0], 1)
        self.assertEqual(snapshot.motion_evidence()[0]["current_center"][0], 2)
        self.assertEqual(snapshot.timestamp, 12.5)
        with self.assertRaises(ValueError):
            snapshot.frame()[0, 0, 0] = 1
        with self.assertRaises(FrozenInstanceError):
            snapshot.timestamp = 99
        decoded = snapshot.motion_evidence()
        decoded.clear()
        self.assertTrue(snapshot.motion_evidence())

    def test_delayed_worker_uses_captured_state(self):
        env = live_functions("analyze_background")
        queued = []
        thread = Mock()
        thread.Thread.side_effect = lambda **kw: Mock(start=lambda: queued.append(kw["target"]))
        consume = Mock()
        env.update(threading=thread, analysis_running=False,
                   analyze_existing_yolo_result=consume)
        frame = np.zeros((1, 1, 3), dtype=np.uint8)
        state = WorldState(10, {"detections": [{"class_name": "person"}]})
        motion = [{"motion_state": "moving_right"}]
        env["analyze_background"](frame, object(), motion_evidence=motion, world_state=state)
        motion[0]["motion_state"] = "moving_left"
        state.data["detections"].clear()
        frame[:] = 255
        queued[0]()
        snapshot = consume.call_args.kwargs["snapshot"]
        self.assertEqual(snapshot.motion_evidence()[0]["motion_state"], "moving_right")
        self.assertTrue(snapshot.world_state().data["detections"])
        self.assertEqual(snapshot.frame().sum(), 0)
        self.assertFalse(env["analysis_running"])

    def test_slow_vlm_cannot_mix_new_motion(self):
        env = live_functions("analyze_existing_yolo_result")
        state = WorldState(7, {"detections": []})
        snapshot = AnalysisSnapshot.capture(np.zeros((2, 2, 3), dtype=np.uint8), state,
                                            [{"motion_state": "moving_right", "normalized_motion": .1}])
        def perceive(path):
            env["latest_stable_motion_evidence"] = [{"motion_state": "moving_left"}]
            return WorldState(99, {})
        import tempfile, os
        coordinator = Mock()
        coordinator.analyze.return_value = {}
        env.update(load_smolvlm=lambda: None, SMOLVLM_MAX_DIMENSION=448,
                   tempfile=tempfile, os=os, cv2=Mock(), smolvlm_lock=threading.Lock(),
                   smolvlm_model=Mock(perceive=perceive),
                   perception_fusion=Mock(fuse=lambda a, b: WorldState(99, {})),
                   orchestrator=coordinator, update_stable_reasoning=Mock())
        env["analyze_existing_yolo_result"](None, None, snapshot=snapshot)
        fused = coordinator.analyze.call_args.args[0]
        self.assertEqual(fused.timestamp, 7)
        self.assertEqual(fused.data["observation_timestamp"], 7)
        self.assertEqual(fused.data["motion_evidence"][0]["motion_state"], "moving_right")


class OcclusionConnectionTests(unittest.TestCase):
    def state(self):
        return WorldState(4, {
            "image_width": 1000, "image_height": 1000,
            "detections": [{"object_id": 0, "class_name": "person", "box_xyxy": [100, 100, 150, 150]}],
            "motion_evidence": [{"class_name": "person", "current_center": [125, 125],
                                 "velocity_x": 5, "velocity_y": 0,
                                 "normalized_motion": .1, "motion_state": "moving_right"}],
            "occlusion_evidence": [{"object_id": 0, "class_name": "person",
                                    "frame_truncated": True, "has_overlap_evidence": False}],
        })

    def test_orchestrator_transfers_context_without_mutating_input(self):
        state = self.state()
        original = copy.deepcopy(state)
        result = AisthesisOrchestrator().analyze(state)
        self.assertEqual(state, original)
        temporal = result["temporal"]
        self.assertEqual(temporal["expected"]["current_occlusion_probability"], .35)
        self.assertTrue(temporal["expected"]["current_occlusion_possible"])
        self.assertEqual(temporal["expected"]["occlusion_prediction"], "no_predicted_occlusion")
        self.assertEqual(temporal["latent"]["latent_temporal_state"], "motion_continuation_possible")
        self.assertTrue(temporal["future"]["current_occlusion_possible"])
        fusion = result["sensor_fusion"]["expected"]
        self.assertEqual(fusion["possible_occluded_object_count"], 1)
        self.assertEqual(fusion["generic_occlusion_probability"], .35)
        self.assertEqual(fusion["active_evidence_sources"], 0)
        self.assertEqual(fusion["fused_hidden_actor_probability"], 0)
        context = temporal["latent"]["occlusion_reasoning"]
        self.assertEqual(set(context["latent"]), {"possible_occluded_objects", "count"})

    def test_legacy_and_malformed_context(self):
        for reasoner in (TemporalPredictionReasoner(), SensorFusionReasoner()):
            state = self.state()
            baseline = reasoner.infer_expected_consequences(state).predictions
            for value in (None, [], {"expected": None}, {"expected": {"occlusion_hypotheses": None}}):
                state.data["occlusion_reasoning"] = value
                self.assertEqual(reasoner.infer_expected_consequences(state).predictions, baseline)

    def test_other_object_context_does_not_attach_to_motion(self):
        state = self.state()
        state.data["occlusion_evidence"][0]["object_id"] = 9
        result = AisthesisOrchestrator().analyze(state)
        self.assertEqual(result["temporal"]["expected"]["current_occlusion_probability"], 0)
        self.assertFalse(result["temporal"]["expected"]["current_occlusion_possible"])

    def test_fusion_consumes_derived_evidence_without_raw_input(self):
        state = self.state()
        result = AisthesisOrchestrator().analyze(state)
        context = result["temporal"]["latent"]["occlusion_reasoning"]
        derived_only = WorldState(data={"occlusion_reasoning": context})
        predictions = SensorFusionReasoner().infer_expected_consequences(derived_only).predictions
        self.assertEqual(predictions["generic_occlusion_probability"], .35)
        self.assertEqual(predictions["possible_occluded_object_count"], 1)
        self.assertEqual(predictions["generic_evidence_sources"], ["occlusion"])
        self.assertEqual(predictions["active_evidence_sources"], 0)

    def test_empty_legacy_world(self):
        result = AisthesisOrchestrator().analyze(WorldState())
        self.assertEqual(result["temporal"]["future"]["predicted_event"], "indeterminate")
        self.assertEqual(result["sensor_fusion"]["expected"]["active_evidence_sources"], 0)


if __name__ == "__main__":
    unittest.main()
