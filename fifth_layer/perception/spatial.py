"""Spatial scene representation for Fifth Layer Engine v0.11."""


def horizontal_position(
    box,
    image_width,
):
    """
    Classify an object's horizontal position.

    Returns:
        "left", "center", or "right"
    """

    left, top, width, height = box

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
    """
    Classify an object's vertical position.

    Returns:
        "top", "middle", or "bottom"
    """

    left, top, width, height = box

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
    """
    Describe the horizontal relation between two objects.

    Returns:
        "left_of", "right_of", or "aligned"
    """

    left_a, top_a, width_a, height_a = box_a
    left_b, top_b, width_b, height_b = box_b

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
    """
    Estimate whether two objects are spatially near or far.

    Uses the distance between object centers,
    normalized by image size.

    Returns:
        "near" or "far"
    """

    left_a, top_a, width_a, height_a = box_a
    left_b, top_b, width_b, height_b = box_b

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

    normalized_distance = (
        distance / image_diagonal
    )

    if normalized_distance < 0.35:
        return "near"

    return "far"
def overlap_relation(
    box_a,
    box_b,
):
    """
    Check whether two object bounding boxes overlap.

    Returns:
        "overlapping" or "separate"
    """

    left_a, top_a, width_a, height_a = box_a
    left_b, top_b, width_b, height_b = box_b

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