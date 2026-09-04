"""Example for Fifth Layer Engine v0.18 occlusion reasoning."""

from fifth_layer.engine import FifthLayerEngine
from fifth_layer.perception.object_detection import ObjectDetectionPerception
from fifth_layer.perception.occlusion import extract_occlusion_evidence
from fifth_layer.reasoners.occlusion import OcclusionReasoner
from fifth_layer.world_state import WorldState


perception = ObjectDetectionPerception(
    model_path="models/yolo11n.onnx",
    confidence_threshold=0.40,
)

perceived_world = perception.perceive(
    "test_image.jpg"
)

occlusion_evidence = extract_occlusion_evidence(
    detections=perceived_world.data["detections"],
    image_width=perceived_world.data["image_width"],
    image_height=perceived_world.data["image_height"],
)

world_state = WorldState(
    timestamp=perceived_world.timestamp,
    data={
        **perceived_world.data,
        "occlusion_evidence": occlusion_evidence,
    },
)

engine = FifthLayerEngine(
    reasoner=OcclusionReasoner()
)

result = engine.step(
    world_state
)

print("EXPECTED CONSEQUENCES")
print(result["expected_consequences"])

print()

print("LATENT STATE")
print(result["latent_state"])

print()

print("FUTURE STATE")
print(result["future_state"])