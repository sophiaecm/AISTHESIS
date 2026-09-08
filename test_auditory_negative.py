"""Negative control test for the AISTHESIS auditory reasoner."""

from fifth_layer.world_state import WorldState
from fifth_layer.reasoners.auditory import AuditoryReasoner


world_state = WorldState(
    data={
        "detections": [
            {
                "class_name": "person",
                "confidence": 0.93,
                "box_xyxy": [100, 100, 300, 400],
            }
        ],
        "motion_evidence": [
            {
                "class_name": "person",
                "motion_state": "stationary",
                "speed_pixels_per_second": 0.0,
            }
        ],
        "scene_description": (
            "A person is standing still in an indoor room."
        ),
    }
)


reasoner = AuditoryReasoner()

expected = reasoner.infer_expected_consequences(
    world_state
)

latent = reasoner.infer_latent_state(
    world_state,
    expected,
)

future = reasoner.infer_future_state(
    latent
)


print("\n=== AISTHESIS AUDITORY NEGATIVE TEST ===\n")

print("Expected auditory consequences:")
print(
    expected.predictions[
        "auditory_consequences"
    ]
)

print("\nLatent auditory state:")
print(latent.features)

print("\nFuture auditory prediction:")
print(future.data)