"""Occlusion evidence extraction for Fifth Layer Engine."""


def _get_xywh(detection):
    """
    Support both legacy [x, y, width, height] boxes
    and new YOLO [x1, y1, x2, y2] boxes.
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
        "Detection must contain either 'box' or 'box_xyxy'."
    )


def extract_occlusion_evidence(
    detections,
    image_width,
    image_height,
):
    """
    Extract observable geometric evidence for possible occlusion.

    This does not claim that an object is truly occluded.
    It only measures frame truncation and bounding-box overlap.
    """

    evidence = []

    for index, detection in enumerate(detections):
        left, top, width, height = _get_xywh(detection)

        right = left + width
        bottom = top + height

        frame_truncated = (
            left <= 0
            or top <= 0
            or right >= image_width
            or bottom >= image_height
        )

        overlapping_objects = []

        for other_index, other_detection in enumerate(detections):
            if index == other_index:
                continue

            (
                other_left,
                other_top,
                other_width,
                other_height,
            ) = _get_xywh(other_detection)

            other_right = other_left + other_width
            other_bottom = other_top + other_height

            overlap_left = max(left, other_left)
            overlap_top = max(top, other_top)
            overlap_right = min(right, other_right)
            overlap_bottom = min(bottom, other_bottom)

            overlap_width = max(
                0.0,
                overlap_right - overlap_left,
            )

            overlap_height = max(
                0.0,
                overlap_bottom - overlap_top,
            )

            overlap_area = overlap_width * overlap_height

            object_area = max(
                width * height,
                1.0,
            )

            overlap_ratio = overlap_area / object_area

            if overlap_ratio > 0:
                overlapping_objects.append(
                    {
                        "object_id": other_index,
                        "class_name": other_detection.get(
                            "class_name",
                            "unknown",
                        ),
                        "overlap_ratio": round(
                            overlap_ratio,
                            3,
                        ),
                    }
                )

        evidence.append(
            {
                "object_id": index,
                "class_name": detection.get(
                    "class_name",
                    "unknown",
                ),
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