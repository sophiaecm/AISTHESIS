"""Example for Fifth Layer Engine v0.7 using SensorFusionReasoner."""

from fifth_layer import (
    FifthLayerEngine,
    SensorFusionReasoner,
    WorldState,
)


reasoner = SensorFusionReasoner()
engine = FifthLayerEngine(reasoner=reasoner)

initial_state = WorldState(
    timestamp=0.0,
    data={
        "vision_hidden_actor_possible": True,
        "vision_confidence": 0.70,
        "motion_toward_occlusion": True,
        "motion_confidence": 0.80,
        "physics_hidden_interaction_possible": True,
        "physics_confidence": 0.90,
    },
)

result = engine.step(initial_state)

print("Expected Consequences:", result["expected_consequences"])
print("Latent State:", result["latent_state"])
print("Future State:", result["future_state"])