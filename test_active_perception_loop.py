from fifth_layer.world_state import WorldState
from fifth_layer.reasoners.semantic_conflict import SemanticConflictReasoner
from fifth_layer.reasoners.active_perception import ActivePerceptionReasoner
from fifth_layer.perception.locate_anything import LocateAnythingPerception


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

semantic_reasoner = SemanticConflictReasoner()
active_reasoner = ActivePerceptionReasoner()

semantic_result = semantic_reasoner.analyze(world_state)

active_query = active_reasoner.build_query(
    semantic_conflict_result=semantic_result,
    scene_description=world_state.data["scene_description"],
)

print("SEMANTIC RESULT:")
print(semantic_result)

print("\nACTIVE QUERY:")
print(active_query)

if active_query["trigger"]:
    locator = LocateAnythingPerception()

    locate_state = locator.perceive(
        "test.jpg",
        targets=active_query["targets"],
        trigger_reason=active_query["reason"],
    )

    print("\nLOCATEANYTHING RESULT:")
    print(locate_state.data)

    locator.close()
else:
    print("\nLocateAnything was not triggered.")