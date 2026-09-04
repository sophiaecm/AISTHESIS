"""Example for Fifth Layer Engine v0.16 visual evidence extraction."""

from fifth_layer.perception.evidence import extract_visual_evidence
from fifth_layer.perception.object_detection import ObjectDetectionPerception


perception = ObjectDetectionPerception(
    model_path="models/yolo11n.onnx",
    confidence_threshold=0.40,
)

world_state = perception.perceive(
    "test_image.jpg"
)

evidence = extract_visual_evidence(
    detections=world_state.data["detections"],
    image_width=world_state.data["image_width"],
    image_height=world_state.data["image_height"],
)

print("VISUAL EVIDENCE")
print()

for item in evidence:
    print(item)