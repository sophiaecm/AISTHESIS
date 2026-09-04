"""Example for Fifth Layer Engine v0.17 occlusion evidence."""

from fifth_layer.perception.object_detection import ObjectDetectionPerception
from fifth_layer.perception.occlusion import extract_occlusion_evidence


perception = ObjectDetectionPerception(
    model_path="models/yolo11n.onnx",
    confidence_threshold=0.40,
)

world_state = perception.perceive(
    "test_image.jpg"
)

occlusion_evidence = extract_occlusion_evidence(
    detections=world_state.data["detections"],
    image_width=world_state.data["image_width"],
    image_height=world_state.data["image_height"],
)

print("OCCLUSION EVIDENCE")
print()

for item in occlusion_evidence:
    print(item)