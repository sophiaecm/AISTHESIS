"""Temporal motion evidence for Fifth Layer Engine v0.19."""


def box_center(box):
    """Return the center point of a bounding box."""

    left, top, width, height = box

    return (
        left + width / 2,
        top + height / 2,
    )


def extract_motion_evidence(
    previous_detections,
    current_detections,
    image_width,
    image_height,
):
    """
    Estimate simple object motion between two frames.

    v0.19 uses class names to create simple object matches.
    This is not yet persistent multi-object tracking.
    """

    evidence = []

    used_current = set()

    image_diagonal = (
        image_width ** 2
        + image_height ** 2
    ) ** 0.5

    for previous_id, previous in enumerate(
        previous_detections
    ):
        best_match_id = None
        best_distance = None

        previous_center = box_center(
            previous["box"]
        )

        for current_id, current in enumerate(
            current_detections
        ):
            if current_id in used_current:
                continue

            if (
                current["class_name"]
                != previous["class_name"]
            ):
                continue

            current_center = box_center(
                current["box"]
            )

            dx = (
                current_center[0]
                - previous_center[0]
            )

            dy = (
                current_center[1]
                - previous_center[1]
            )

            distance = (
                dx ** 2
                + dy ** 2
            ) ** 0.5

            if (
                best_distance is None
                or distance < best_distance
            ):
                best_distance = distance
                best_match_id = current_id

        if best_match_id is None:
            continue

        used_current.add(
            best_match_id
        )

        current = current_detections[
            best_match_id
        ]

        current_center = box_center(
            current["box"]
        )

        dx = (
            current_center[0]
            - previous_center[0]
        )

        dy = (
            current_center[1]
            - previous_center[1]
        )

        normalized_motion = (
            best_distance / image_diagonal
            if image_diagonal > 0
            else 0.0
        )

        if normalized_motion < 0.01:
            motion_state = "stationary"

        elif abs(dx) >= abs(dy):
            if dx > 0:
                motion_state = "moving_right"
            else:
                motion_state = "moving_left"

        else:
            if dy > 0:
                motion_state = "moving_down"
            else:
                motion_state = "moving_up"

        evidence.append(
            {
                "previous_object_id": previous_id,
                "current_object_id": best_match_id,
                "class_name": current[
                    "class_name"
                ],
                "dx": round(dx, 3),
                "dy": round(dy, 3),
                "normalized_motion": round(
                    normalized_motion,
                    4,
                ),
                "motion_state": motion_state,
            }
        )

    return evidence