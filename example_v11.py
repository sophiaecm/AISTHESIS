"""Example for Fifth Layer Engine v0.11 spatial reasoning."""

from fifth_layer.perception.object_detection import ObjectDetectionPerception
from fifth_layer.perception.spatial import (
    horizontal_position,
    vertical_position,
    horizontal_relation,
    distance_relation,
    overlap_relation,
)


perception = ObjectDetectionPerception(
    model_path="models/yolo11n.onnx",
    confidence_threshold=0.40,
)

world_state = perception.perceive("test_image.jpg")

image_width = world_state.data["image_width"]
image_height = world_state.data["image_height"]
detections = world_state.data["detections"]

for detection in detections:
    horizontal = horizontal_position(
        detection["box"],
        image_width,
    )

    vertical = vertical_position(
        detection["box"],
        image_height,
    )

    print(
        detection["class_name"],
        "→",
        horizontal,
        "/",
        vertical,
    )

if len(detections) >= 2:
    first = detections[0]
    second = detections[1]

    relation = horizontal_relation(
        first["box"],
        second["box"],
    )

    distance = distance_relation(
        first["box"],
        second["box"],
        image_width,
        image_height,
    )

    overlap = overlap_relation(
        first["box"],
        second["box"],
    )

    print(
        first["class_name"],
        relation,
        second["class_name"],
    )

    print(
        first["class_name"],
        distance,
        second["class_name"],
    )

    print(
        first["class_name"],
        overlap,
        second["class_name"],
    )