"""Example for Fifth Layer Engine v0.4 using RuleBasedReasoner."""

from fifth_layer import (
    FifthLayerEngine,
    RuleBasedReasoner,
    WorldState,
)


reasoner = RuleBasedReasoner()
engine = FifthLayerEngine(reasoner=reasoner)

initial_state = WorldState(
    timestamp=0.0,
    data={
        "object_type": "ball",
        "object_motion": "toward_occlusion",
        "occlusion": True,
    },
)

result = engine.step(initial_state)

print("Expected Consequences:", result["expected_consequences"])
print("Latent State:", result["latent_state"])
print("Future State:", result["future_state"])
