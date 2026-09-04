"""Example for Fifth Layer Engine v0.12 scene graph."""

from fifth_layer.perception.object_detection import ObjectDetectionPerception
from fifth_layer.perception.scene_graph import build_scene_graph


perception = ObjectDetectionPerception(
    model_path="models/yolo11n.onnx",
    confidence_threshold=0.40,
)

world_state = perception.perceive("test_image.jpg")

scene_graph = build_scene_graph(
    detections=world_state.data["detections"],
    image_width=world_state.data["image_width"],
    image_height=world_state.data["image_height"],
)

print("NODES")

for node in scene_graph.nodes:
    print(node)

print()

print("RELATIONS")

for relation in scene_graph.relations:
    print(relation)