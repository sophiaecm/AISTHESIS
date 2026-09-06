from fifth_layer.reasoners.active_perception import ActivePerceptionReasoner


semantic_conflict_result = {
    "semantic_conflict_detected": True,
    "semantic_conflicts": [
        {
            "detected_class": "toothbrush",
            "confidence": 0.42,
            "reason": "detector_class_not_supported_by_scene_description",
        }
    ],
    "should_trigger_active_perception": True,
}

scene_description = (
    "A person is holding a small medicine package."
)

reasoner = ActivePerceptionReasoner()

result = reasoner.build_query(
    semantic_conflict_result=semantic_conflict_result,
    scene_description=scene_description,
)

print(result)