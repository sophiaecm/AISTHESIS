"""Spatial scene representation for Fifth Layer Engine."""


def _to_xywh(box):
    """
    Accept either:
    - [left, top, width, height]
    - [x1, y1, x2, y2] wrapped as {"box_xyxy": [...]}
    """

    if isinstance(box, dict):
        if "box_xyxy" in box:
            x1, y1, x2, y2 = box["box_xyxy"]
            return (
                float(x1),
                float(y1),
                float(x2 - x1),
                float(y2 - y1),
            )

        if "box" in box:
            left, top, width, height = box["box"]
            return (
                float(left),
                float(top),
                float(width),
                float(height),
            )

    left, top, width, height = box

    return (
        float(left),
        float(top),
        float(width),
        float(height),
    )


def horizontal_position(
    box,
    image_width,
):
    left, top, width, height = _to_xywh(box)

    object_center_x = left + width / 2

    left_boundary = image_width / 3
    right_boundary = image_width * 2 / 3

    if object_center_x < left_boundary:
        return "left"

    if object_center_x > right_boundary:
        return "right"

    return "center"


def vertical_position(
    box,
    image_height,
):
    left, top, width, height = _to_xywh(box)

    object_center_y = top + height / 2

    top_boundary = image_height / 3
    bottom_boundary = image_height * 2 / 3

    if object_center_y < top_boundary:
        return "top"

    if object_center_y > bottom_boundary:
        return "bottom"

    return "middle"


def horizontal_relation(
    box_a,
    box_b,
):
    left_a, top_a, width_a, height_a = _to_xywh(box_a)
    left_b, top_b, width_b, height_b = _to_xywh(box_b)

    center_a = left_a + width_a / 2
    center_b = left_b + width_b / 2

    if center_a < center_b:
        return "left_of"

    if center_a > center_b:
        return "right_of"

    return "aligned"


def distance_relation(
    box_a,
    box_b,
    image_width,
    image_height,
):
    left_a, top_a, width_a, height_a = _to_xywh(box_a)
    left_b, top_b, width_b, height_b = _to_xywh(box_b)

    center_a_x = left_a + width_a / 2
    center_a_y = top_a + height_a / 2

    center_b_x = left_b + width_b / 2
    center_b_y = top_b + height_b / 2

    dx = center_a_x - center_b_x
    dy = center_a_y - center_b_y

    distance = (dx ** 2 + dy ** 2) ** 0.5

    image_diagonal = (
        image_width ** 2
        + image_height ** 2
    ) ** 0.5

    normalized_distance = distance / image_diagonal

    if normalized_distance < 0.35:
        return "near"

    return "far"


def overlap_relation(
    box_a,
    box_b,
):
    left_a, top_a, width_a, height_a = _to_xywh(box_a)
    left_b, top_b, width_b, height_b = _to_xywh(box_b)

    right_a = left_a + width_a
    bottom_a = top_a + height_a

    right_b = left_b + width_b
    bottom_b = top_b + height_b

    overlap_x = (
        left_a < right_b
        and right_a > left_b
    )

    overlap_y = (
        top_a < bottom_b
        and bottom_a > top_b
    )

    if overlap_x and overlap_y:
        return "overlapping"

    return "separate"