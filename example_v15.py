"""Example for Fifth Layer Engine v0.15 composite reasoning."""

from fifth_layer.engine import FifthLayerEngine
from fifth_layer.perception.object_detection import ObjectDetectionPerception
from fifth_layer.perception.scene_graph import build_scene_graph
from fifth_layer.reasoners.composite import CompositeReasoner
from fifth_layer.reasoners.physics import PhysicsReasoner
from fifth_layer.reasoners.probabilistic import ProbabilisticReasoner
from fifth_layer.reasoners.scene import SceneReasoner
from fifth_layer.reasoners.sensor_fusion import SensorFusionReasoner
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

        # Simple test values for the other reasoners.
        "position": (0.0, 0.0),
        "velocity": (1.0, 0.0),
        "dt": 1.0,
        "occlusion_zone": (
            0.5,
            -1.0,
            2.0,
            1.0,
        ),

        "occlusion": True,
        "object_motion": "toward_occlusion",
        "object_type": "ball",
        "trajectory_continues": True,

        "vision_confidence": 0.70,
        "motion_confidence": 0.80,
        "physics_confidence": 0.90,
        "vision_hidden_actor_possible": True,
        "motion_toward_occlusion": True,
        "physics_hidden_interaction_possible": True,
    },
)

reasoner = CompositeReasoner(
    reasoners={
        "scene": SceneReasoner(),
        "physics": PhysicsReasoner(),
        "probabilistic": ProbabilisticReasoner(),
        "sensor_fusion": SensorFusionReasoner(),
    }
)

engine = FifthLayerEngine(
    reasoner=reasoner
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