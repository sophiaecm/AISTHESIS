"""Example for Fifth Layer Engine v0.13 local scene description."""

from fifth_layer.perception.object_detection import ObjectDetectionPerception
from fifth_layer.perception.scene_graph import build_scene_graph
from fifth_layer.perception.scene_description import describe_scene


perception = ObjectDetectionPerception(
    model_path="models/yolo11n.onnx",
    confidence_threshold=0.40,
)

world_state = perception.perceive(
    "test_image.jpg"
)

scene_graph = build_scene_graph(
    detections=world_state.data["detections"],
    image_width=world_state.data["image_width"],
    image_height=world_state.data["image_height"],
)

description = describe_scene(
    scene_graph
)

print("SCENE DESCRIPTION")
print()
print(description)