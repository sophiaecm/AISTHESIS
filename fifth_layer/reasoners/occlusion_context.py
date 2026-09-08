"""Optional occlusion output contract shared by downstream reasoners."""

from math import isfinite


def occlusion_context(data):
    context = data.get("occlusion_reasoning")
    if not isinstance(context, dict):
        return None
    expected = context.get("expected", {})
    latent = context.get("latent", {})
    if not isinstance(expected, dict) or not isinstance(latent, dict):
        return None
    hypotheses = expected.get("occlusion_hypotheses", [])
    objects = latent.get("possible_occluded_objects", [])
    if not isinstance(hypotheses, list) or not isinstance(objects, list):
        return None
    valid = []
    for item in hypotheses:
        if not isinstance(item, dict):
            continue
        probability = item.get("occlusion_probability")
        if (isinstance(probability, (int, float)) and isfinite(probability)
                and 0 <= probability <= 1):
            valid.append(item)
    # Latent membership must be supported by an expected hypothesis.
    possible = [item for item in valid
                if item.get("hypothesis") in {"occlusion_possible", "occlusion_likely"}
                and any(isinstance(obj, dict)
                        and obj.get("object_id") == item.get("object_id")
                        and obj.get("class_name") == item.get("class_name")
                        for obj in objects)]
    return valid, possible
