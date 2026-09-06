from fifth_layer.world_state import WorldState
from fifth_layer.reasoners.semantic_conflict import SemanticConflictReasoner


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
    },
)

reasoner = SemanticConflictReasoner()

result = reasoner.analyze(world_state)

print(result)