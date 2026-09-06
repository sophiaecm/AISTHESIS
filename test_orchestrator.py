from fifth_layer.world_state import WorldState
from fifth_layer.reasoners.orchestrator import AisthesisOrchestrator


world_state = WorldState(
    timestamp=0.0,
    data={
        "detections": [
            {
                "class_name": "toothbrush",
                "confidence": 0.42,
            }
        ],
        "scene_description": (
            "A person is holding a small medicine package."
        ),
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
        ],
    },
)

orchestrator = AisthesisOrchestrator()

result = orchestrator.analyze(world_state)

print("AISTHESIS ORCHESTRATOR RESULT:")
print(result)