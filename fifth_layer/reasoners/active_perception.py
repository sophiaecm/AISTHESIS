class ActivePerceptionReasoner:
    """
    Decide when targeted visual search should be triggered
    and which hypotheses should be verified.
    """

    def build_query(
        self,
        semantic_conflict_result: dict,
        scene_description: str = "",
    ) -> dict:
        if not semantic_conflict_result.get(
            "should_trigger_active_perception",
            False,
        ):
            return {
                "trigger": False,
                "targets": [],
                "reason": "no_semantic_conflict",
            }

        targets = []

        for conflict in semantic_conflict_result.get(
            "semantic_conflicts",
            [],
        ):
            detected_class = conflict.get("detected_class")

            if detected_class:
                targets.append(detected_class)

        description_lower = scene_description.lower()

        candidate_terms = [
            "medicine box",
            "medicine package",
            "bottle",
            "phone",
            "cup",
            "book",
            "plant",
            "person",
        ]

        for candidate in candidate_terms:
            if candidate in description_lower:
                targets.append(candidate)

        targets = list(dict.fromkeys(targets))

        return {
            "trigger": True,
            "targets": targets,
            "reason": "semantic_conflict_requires_targeted_visual_search",
        }