from fifth_layer.world_state import WorldState
from fifth_layer.reasoners.occlusion import OcclusionReasoner


world_state = WorldState(
    timestamp=0.0,
    data={
        "occlusion_evidence": [
            {
                "object_id": 0,
                "class_name": "person",
                "frame_truncated": False,
                "overlapping_objects": [
                    {
                        "object_id": 1,
                        "class_name": "box",
                        "overlap_ratio": 0.2,
                    }
                ],
                "has_overlap_evidence": True,
                "possible_occlusion_evidence": True,
            },
            {
                "object_id": 1,
                "class_name": "box",
                "frame_truncated": False,
                "overlapping_objects": [
                    {
                        "object_id": 0,
                        "class_name": "person",
                        "overlap_ratio": 0.571,
                    }
                ],
                "has_overlap_evidence": True,
                "possible_occlusion_evidence": True,
            },
        ]
    },
)

reasoner = OcclusionReasoner()

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

print("EXPECTED:")
print(expected.predictions)

print("\nLATENT:")
print(latent.features)

print("\nFUTURE:")
print(future.data)