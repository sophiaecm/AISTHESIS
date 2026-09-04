"""Example for Fifth Layer Engine v0.10 using ObjectDetectionPerception."""

from fifth_layer.perception.object_detection import ObjectDetectionPerception


perception = ObjectDetectionPerception(
    model_path="models/yolo11n.onnx",
    confidence_threshold=0.40,
)

world_state = perception.perceive("test_image.jpg")

print(world_state)