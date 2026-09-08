from fifth_layer.world_state import WorldState
from fifth_layer.perception.temporal import extract_motion_evidence
from fifth_layer.reasoners.temporal_prediction import (
    TemporalPredictionReasoner,
)


# ---------------------------------------------------------
# Previous frame
# ---------------------------------------------------------

previous_detections = [
    {
        "object_id": 0,
        "class_name": "person",
        "confidence": 0.90,
        "box_xyxy": [
            100,
            100,
            300,
            400,
        ],
    }
]


# ---------------------------------------------------------
# Current frame
#
# Person moved right.
# Chair is positioned further along that trajectory.
# ---------------------------------------------------------

current_detections = [
    {
        "object_id": 0,
        "class_name": "person",
        "confidence": 0.91,
        "box_xyxy": [
            140,
            100,
            340,
            400,
        ],
    },
    {
        "object_id": 1,
        "class_name": "chair",
        "confidence": 0.88,
        "box_xyxy": [
            390,
            120,
            520,
            420,
        ],
    },
]


# ---------------------------------------------------------
# Temporal perception
# ---------------------------------------------------------

motion_evidence = extract_motion_evidence(
    previous_detections=previous_detections,
    current_detections=current_detections,
    image_width=640,
    image_height=480,
    delta_time=0.5,
)


print()
print("TEMPORAL MOTION EVIDENCE")
print("------------------------")

for item in motion_evidence:
    print(item)


# ---------------------------------------------------------
# Build WorldState
# ---------------------------------------------------------

world_state = WorldState(
    timestamp=0.0,
    data={
        "source_type": "test",
        "image_width": 640,
        "image_height": 480,
        "detections": current_detections,
        "motion_evidence": motion_evidence,
    },
)


# ---------------------------------------------------------
# Trajectory + occlusion-aware reasoning
# ---------------------------------------------------------

reasoner = TemporalPredictionReasoner()

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


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print()
print("TRAJECTORY")
print("----------")

trajectory = expected.predictions.get(
    "trajectory",
    [],
)

for point in trajectory:
    print(point)


print()
print("TRAJECTORY SUMMARY")
print("------------------")

print(
    "Predicted center 1s:",
    expected.predictions.get(
        "predicted_center_1s"
    ),
)

print(
    "Visibility:",
    expected.predictions.get(
        "visibility_prediction"
    ),
)

print(
    "Occlusion:",
    expected.predictions.get(
        "occlusion_prediction"
    ),
)

print(
    "Potential occluder:",
    expected.predictions.get(
        "potential_occluder"
    ),
)


print()
print("LATENT STATE")
print("------------")

print(
    latent.features.get(
        "latent_temporal_state"
    )
)


print()
print("FUTURE PREDICTION")
print("-----------------")

print(
    future.data.get(
        "predicted_event"
    )
)

print(
    "Future visibility:",
    future.data.get(
        "future_visibility_state"
    ),
)