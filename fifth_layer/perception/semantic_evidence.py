import re


def extract_semantic_evidence(scene_description: str) -> dict:
    """
    Convert a natural-language scene description into
    lightweight structured semantic evidence.

    This is not object detection and does not create bounding boxes.
    It only records concepts explicitly supported by the description.
    """

    text = (scene_description or "").strip().lower()

    evidence = {
        "description_available": bool(text),
        "mentioned_objects": [],
        "mentioned_actions": [],
        "mentioned_relations": [],
        "semantic_text": scene_description or "",
    }

    if not text:
        return evidence

    object_terms = [
        "person",
        "people",
        "plant",
        "flame",
        "fire",
        "phone",
        "cell phone",
        "medicine",
        "medicine package",
        "medicine box",
        "bottle",
        "box",
        "book",
        "car",
        "vehicle",
        "bicycle",
        "dog",
        "cat",
        "chair",
        "table",
        "computer",
        "laptop",
        "toothbrush",
        "cup",
        "ball",
    ]

    action_terms = [
        "holding",
        "carrying",
        "touching",
        "using",
        "walking",
        "running",
        "standing",
        "sitting",
        "moving",
        "watering",
        "approaching",
        "falling",
        "placing",
        "grabbing",
    ]

    relation_terms = [
        "near",
        "next to",
        "behind",
        "in front of",
        "above",
        "below",
        "inside",
        "outside",
        "overlapping",
        "under",
        "beside",
    ]

    mentioned_objects = []

    for term in object_terms:
        if re.search(
            r"\b" + re.escape(term) + r"\b",
            text,
        ):
            mentioned_objects.append(term)

    mentioned_actions = []

    for term in action_terms:
        if re.search(
            r"\b" + re.escape(term) + r"\b",
            text,
        ):
            mentioned_actions.append(term)

    mentioned_relations = []

    for term in relation_terms:
        if term in text:
            mentioned_relations.append(term)

    evidence["mentioned_objects"] = list(
        dict.fromkeys(mentioned_objects)
    )

    evidence["mentioned_actions"] = list(
        dict.fromkeys(mentioned_actions)
    )

    evidence["mentioned_relations"] = list(
        dict.fromkeys(mentioned_relations)
    )

    return evidence