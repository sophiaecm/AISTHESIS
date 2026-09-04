"""Example for Fifth Layer Engine v0.14 scene-aware reasoning."""

from fifth_layer.engine import FifthLayerEngine
from fifth_layer.perception.object_detection import ObjectDetectionPerception
from fifth_layer.perception.scene_graph import build_scene_graph
from fifth_layer.reasoners.scene import SceneReasoner
from fifth_layer.world_state import WorldState


perception = ObjectDetectionPerception(
    model_path="models/yolo11n.onnx",
    confidence_threshold=0.40,
)

perceived_world = perception.perceive(
    "test_image.jpg"
)

scene_graph = build_scene_graph(
    detections=perceived_world.data["detections"],
    image_width=perceived_world.data["image_width"],
    image_height=perceived_world.data["image_height"],
)

scene_relations = []

for relation in scene_graph.relations:
    scene_relations.append(
        {
            "source_id": relation.source_id,
            "relation": relation.relation,
            "target_id": relation.target_id,
        }
    )

world_state = WorldState(
    timestamp=perceived_world.timestamp,
    data={
        **perceived_world.data,
        "scene_relations": scene_relations,
    },
)

engine = FifthLayerEngine(
    reasoner=SceneReasoner()
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