"""Lazy adapters around unchanged repository implementations."""
from copy import deepcopy
from pathlib import Path
import ast
import inspect


class RepositoryAdapters:
    """One instance per mode process; models load only on the requested path."""
    def __init__(self, config):
        self.config = dict(config)
        self.detector_model = None
        self.language_model = None
        self.tracker = None
        self.previous = None
        self.previous_timestamp = None

    def detector(self, path):
        if self.detector_model is None:
            from fifth_layer.perception.yolo_detector import YoloDetectorPerception
            model = Path(self.config['detector_model']).resolve(strict=True)
            self.detector_model = YoloDetectorPerception(str(model),
                confidence_threshold=self.config.get('detector_confidence', .4),
                device=self.config.get('detector_device', 'cpu'))
        return self.detector_model.perceive(path)

    def vlm(self, path):
        if self.language_model is None:
            # Require a pre-existing local snapshot. Workers set offline flags too.
            model = Path(self.config['vlm_model']).resolve(strict=True)
            if not model.is_dir():
                raise ValueError('vlm_model must be an existing local model directory')
            from fifth_layer.perception.smolvlm_scene import SmolVLMScenePerception
            self.language_model = SmolVLMScenePerception(str(model), device=self.config.get('vlm_device', 'cpu'))
        return self.language_model.perceive(path)

    def structured(self, raw, timestamp):
        from fifth_layer.world_state import WorldState
        from fifth_layer.perception.tracking import TrackMemory
        from fifth_layer.perception.temporal import extract_motion_evidence
        from fifth_layer.perception.occlusion import extract_occlusion_evidence
        from fifth_layer.perception.perception_fusion import PerceptionFusion
        if self.tracker is None:
            self.tracker = TrackMemory()  # documented legacy defaults, fresh per mode
        width, height = raw.data['image_width'], raw.data['image_height']
        detections = self.tracker.update(deepcopy(raw.data['detections']), timestamp, width, height)
        motion = extract_motion_evidence(self.previous or [], detections, width, height,
            delta_time=None if self.previous_timestamp is None else timestamp - self.previous_timestamp)
        self.previous, self.previous_timestamp = deepcopy(detections), timestamp
        return WorldState(timestamp, dict(image_width=width, image_height=height,
            detections=detections, accepted_detections=deepcopy(detections), detection_count=len(detections),
            predicted_tracks=deepcopy(self.tracker.predicted_tracks), motion_evidence=motion,
            scene_relations=PerceptionFusion()._build_scene_relations(detections, width, height),
            occlusion_evidence=extract_occlusion_evidence(detections, width, height)))

    def combine(self, structured, vlm):
        from fifth_layer.perception.perception_fusion import PerceptionFusion
        combined = PerceptionFusion().fuse(deepcopy(structured), deepcopy(vlm))
        # Match existing offline/live orchestration: observation time and motion
        # are carried alongside fusion's detector/semantic/spatial outputs.
        combined.timestamp = structured.timestamp
        combined.data['motion_evidence'] = deepcopy(structured.data['motion_evidence'])
        return combined

    def fifth(self, state):
        from fifth_layer.reasoners.orchestrator import AisthesisOrchestrator
        return AisthesisOrchestrator().analyze(deepcopy(state))

    def metadata(self):
        result = dict(configuration=self.config, tracking_policy='TrackMemory defaults; fresh per condition',
                      baseline='detector_tracking_geometry_v1', image_resize='none',
                      combined_path='PerceptionFusion.fuse -> AisthesisOrchestrator.analyze')
        if self.language_model is not None:
            method = self.language_model.perceive
            source = inspect.getsource(method)
            # Record exact string constants, including the original prompt,
            # without replacing or intercepting the production call.
            import textwrap
            constants = [node.value for node in ast.walk(ast.parse(textwrap.dedent(source)))
                         if isinstance(node, ast.Constant) and isinstance(node.value, str)]
            result['vlm'] = dict(model=self.language_model.model_id, device=self.language_model.device,
                perceive_source=source, source_string_constants=constants,
                generation='existing implementation; do_sample=False, max_new_tokens=128',
                stochasticity='No sampling requested; hardware/library nondeterminism may remain.')
        return result
