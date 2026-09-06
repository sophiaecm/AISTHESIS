from fifth_layer.world_state import WorldState


class SemanticConflictReasoner:
    """
    Detect semantic disagreement between object detection
    and scene-level language description.
    """

    def analyze(self, world_state: WorldState) -> dict:
        data = world_state.data

        detections = data.get("detections", [])
        scene_description = data.get("scene_description", "")

        description_lower = scene_description.lower()

        conflicts = []

        for detection in detections:
            class_name = detection.get("class_name", "")
            confidence = detection.get("confidence", 0.0)

            if not class_name:
                continue

            class_lower = class_name.lower()

            if class_lower not in description_lower:
                conflicts.append(
                    {
                        "detected_class": class_name,
                        "confidence": confidence,
                        "reason": "detector_class_not_supported_by_scene_description",
                    }
                )

        conflict_detected = len(conflicts) > 0

        return {
            "semantic_conflict_detected": conflict_detected,
            "semantic_conflicts": conflicts,
            "should_trigger_active_perception": conflict_detected,
        }