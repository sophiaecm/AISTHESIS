from fifth_layer.perception.yolo_detector import YoloDetectorPerception
from fifth_layer.perception.smolvlm_scene import SmolVLMScenePerception
from fifth_layer.perception.perception_fusion import PerceptionFusion

yolo = YoloDetectorPerception(
    model_path="yolo11n.pt",
    confidence_threshold=0.20,
    device=0,
)

smolvlm = SmolVLMScenePerception(
    model_id="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    device="cuda",
)

yolo_state = yolo.perceive("test.jpg")
smolvlm_state = smolvlm.perceive("test.jpg")

fusion = PerceptionFusion()

fused_state = fusion.fuse(
    yolo_state=yolo_state,
    smolvlm_state=smolvlm_state,
)

print(fused_state)
print(fused_state.data)