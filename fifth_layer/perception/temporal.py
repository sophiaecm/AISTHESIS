"""Temporal motion evidence for Fifth Layer Engine."""

import math


def _box_to_xywh(detection):
    """
    Convert supported detection box formats to
    (left, top, width, height).

    Supports:
    - box_xyxy: [x1, y1, x2, y2]
    - box: [left, top, width, height]
    """

    if "box_xyxy" in detection:
        x1, y1, x2, y2 = detection["box_xyxy"]

        return (
            float(x1),
            float(y1),
            float(x2 - x1),
            float(y2 - y1),
        )

    if "box" in detection:
        left, top, width, height = detection["box"]

        return (
            float(left),
            float(top),
            float(width),
            float(height),
        )

    raise ValueError(
        "Detection must contain either 'box_xyxy' or 'box'."
    )


def box_center(detection):
    """Return the center point of a detection."""

    left, top, width, height = _box_to_xywh(
        detection
    )

    return (
        left + width / 2.0,
        top + height / 2.0,
    )


def extract_motion_evidence(
    previous_detections,
    current_detections,
    image_width,
    image_height,
    delta_time=None,
):
    """
    Estimate object motion between two frames.

    Objects are matched using:
    1. Same class
    2. Nearest center
    3. Maximum normalized matching distance

    This is lightweight temporal association rather than
    full persistent multi-object tracking.
    """

    evidence = []

    if not previous_detections:
        return evidence

    if not current_detections:
        return evidence

    image_diagonal = math.sqrt(
        image_width ** 2
        + image_height ** 2
    )

    if image_diagonal <= 0:
        return evidence

    used_current = set()

    # Prevent an object from being matched to another
    # same-class object on the opposite side of the frame.
    maximum_match_distance = 0.20

    for previous_id, previous in enumerate(
        previous_detections
    ):
        previous_class = previous.get(
            "class_name",
            "unknown",
        )

        previous_center = box_center(
            previous
        )

        best_match_id = None
        best_distance = None

        for current_id, current in enumerate(
            current_detections
        ):
            if current_id in used_current:
                continue

            if "track_id" in previous or "track_id" in current:
                if previous.get("track_id") != current.get("track_id"):
                    continue


            current_class = current.get(
                "class_name",
                "unknown",
            )

            if current_class != previous_class:
                continue

            current_center = box_center(
                current
            )

            dx = (
                current_center[0]
                - previous_center[0]
            )

            dy = (
                current_center[1]
                - previous_center[1]
            )

            distance = math.sqrt(
                dx ** 2
                + dy ** 2
            )

            normalized_distance = (
                distance
                / image_diagonal
            )

            if (
                normalized_distance
                > maximum_match_distance
            ):
                continue

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
            current
        )

        dx = (
            current_center[0]
            - previous_center[0]
        )

        dy = (
            current_center[1]
            - previous_center[1]
        )

        pixel_distance = math.sqrt(
            dx ** 2
            + dy ** 2
        )

        normalized_motion = (
            pixel_distance
            / image_diagonal
        )

        if normalized_motion < 0.005:
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

        velocity_x = None
        velocity_y = None
        speed = None

        if (
            delta_time is not None
            and delta_time > 0
        ):
            velocity_x = dx / delta_time
            velocity_y = dy / delta_time

            speed = math.sqrt(
                velocity_x ** 2
                + velocity_y ** 2
            )

        evidence.append(
            {
                **({"track_id": current["track_id"]} if "track_id" in current else {}),
                "previous_object_id": previous_id,
                "current_object_id": best_match_id,

                "class_name": current.get(
                    "class_name",
                    "unknown",
                ),

                "previous_center": [
                    round(
                        previous_center[0],
                        2,
                    ),
                    round(
                        previous_center[1],
                        2,
                    ),
                ],

                "current_center": [
                    round(
                        current_center[0],
                        2,
                    ),
                    round(
                        current_center[1],
                        2,
                    ),
                ],

                "dx": round(
                    dx,
                    3,
                ),

                "dy": round(
                    dy,
                    3,
                ),

                "normalized_motion": round(
                    normalized_motion,
                    4,
                ),

                "motion_state": motion_state,

                "velocity_x": (
                    round(
                        velocity_x,
                        3,
                    )
                    if velocity_x is not None
                    else None
                ),

                "velocity_y": (
                    round(
                        velocity_y,
                        3,
                    )
                    if velocity_y is not None
                    else None
                ),

                "speed_pixels_per_second": (
                    round(
                        speed,
                        3,
                    )
                    if speed is not None
                    else None
                ),
            }
        )

    return evidence