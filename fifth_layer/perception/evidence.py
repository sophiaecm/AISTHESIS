"""Visual evidence extraction for Fifth Layer Engine v0.16."""


def extract_visual_evidence(
    detections,
    image_width,
    image_height,
):
    """
    Convert object detections into simple structured visual evidence.

    This first version extracts:
    - normalized object size
    - border proximity
    - coarse scene occupancy
    """

    evidence = []

    image_area = (
        image_width * image_height
    )

    for index, detection in enumerate(detections):
        left, top, width, height = detection["box"]

        object_area = (
            max(width, 0)
            * max(height, 0)
        )

        area_ratio = (
            object_area / image_area
            if image_area > 0
            else 0.0
        )

        right = left + width
        bottom = top + height

        touches_left = left <= 0
        touches_top = top <= 0
        touches_right = right >= image_width
        touches_bottom = bottom >= image_height

        near_border = (
            touches_left
            or touches_top
            or touches_right
            or touches_bottom
        )

        if area_ratio >= 0.30:
            size_category = "large"

        elif area_ratio >= 0.08:
            size_category = "medium"

        else:
            size_category = "small"

        evidence.append(
            {
                "object_id": index,
                "class_name": detection["class_name"],
                "confidence": detection["confidence"],
                "area_ratio": round(
                    area_ratio,
                    3,
                ),
                "size_category": size_category,
                "near_image_border": near_border,
                "touches_left_border": touches_left,
                "touches_top_border": touches_top,
                "touches_right_border": touches_right,
                "touches_bottom_border": touches_bottom,
            }
        )

    return evidence