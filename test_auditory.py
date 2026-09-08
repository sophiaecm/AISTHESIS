"""Test the AISTHESIS auditory consequence reasoner."""

from fifth_layer.world_state import WorldState
from fifth_layer.reasoners.auditory import AuditoryReasoner


world_state = WorldState(
    data={
        "detections": [
            {
                "class_name": "person",
                "confidence": 0.91,
                "box_xyxy": [100, 100, 300, 400],
            }
        ],
        "motion_evidence": [
            {
                "class_name": "person",
                "motion_state": "moving_right",
                "speed_pixels_per_second": 120.0,
            }
        ],
        "scene_description": (
            "A person is walking through an indoor room."
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


print("\n=== AISTHESIS AUDITORY TEST ===\n")

print("Observed auditory modality:")
print(
    expected.predictions[
        "auditory_modality_observed"
    ]
)

print("\nExpected auditory consequences:")

for consequence in expected.predictions[
    "auditory_consequences"
]:
    print(
        f"- {consequence['consequence']}: "
        f"{consequence['probability']:.2f} "
        f"(uncertainty "
        f"{consequence['uncertainty']:.2f})"
    )

print("\nLatent auditory state:")
print(latent.features)

print("\nFuture auditory prediction:")
print(future.data)