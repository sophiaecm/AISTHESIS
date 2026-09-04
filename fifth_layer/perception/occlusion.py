"""Occlusion evidence extraction for Fifth Layer Engine v0.17."""


def extract_occlusion_evidence(
    detections,
    image_width,
    image_height,
):
    """
    Extract geometric evidence that may indicate partial occlusion.

    Important:
    This module does not claim that an object is truly occluded.
    It only produces observable geometric evidence.
    """

    evidence = []

    for index, detection in enumerate(detections):
        left, top, width, height = detection["box"]

        right = left + width
        bottom = top + height

        frame_truncated = (
            left <= 0
            or top <= 0
            or right >= image_width
            or bottom >= image_height
        )

        overlapping_objects = []

        for other_index, other_detection in enumerate(
            detections
        ):
            if index == other_index:
                continue

            other_left, other_top, other_width, other_height = (
                other_detection["box"]
            )

            other_right = (
                other_left + other_width
            )
            other_bottom = (
                other_top + other_height
            )

            overlap_left = max(
                left,
                other_left,
            )
            overlap_top = max(
                top,
                other_top,
            )
            overlap_right = min(
                right,
                other_right,
            )
            overlap_bottom = min(
                bottom,
                other_bottom,
            )

            overlap_width = max(
                0,
                overlap_right - overlap_left,
            )

            overlap_height = max(
                0,
                overlap_bottom - overlap_top,
            )

            overlap_area = (
                overlap_width
                * overlap_height
            )

            object_area = max(
                width * height,
                1,
            )

            overlap_ratio = (
                overlap_area / object_area
            )

            if overlap_ratio > 0:
                overlapping_objects.append(
                    {
                        "object_id": other_index,
                        "class_name": other_detection[
                            "class_name"
                        ],
                        "overlap_ratio": round(
                            overlap_ratio,
                            3,
                        ),
                    }
                )

        evidence.append(
            {
                "object_id": index,
                "class_name": detection[
                    "class_name"
                ],
                "frame_truncated": frame_truncated,
                "overlapping_objects": overlapping_objects,
                "has_overlap_evidence": bool(
                    overlapping_objects
                ),
                "possible_occlusion_evidence": (
                    frame_truncated
                    or bool(overlapping_objects)
                ),
            }
        )

    return evidence