"""Example for Fifth Layer Engine v0.5 using PhysicsReasoner."""

from fifth_layer import (
    FifthLayerEngine,
    PhysicsReasoner,
    WorldState,
)


reasoner = PhysicsReasoner()
engine = FifthLayerEngine(reasoner=reasoner)

initial_state = WorldState(
    timestamp=0.0,
    data={
        "position": (1.0, 1.0),
        "velocity": (2.0, 0.5),
        "dt": 1.0,
        "occlusion_zone": (2.5, 1.0, 4.0, 2.0),
    },
)

result = engine.step(initial_state)

print("Expected Consequences:", result["expected_consequences"])
print("Latent State:", result["latent_state"])
print("Future State:", result["future_state"])