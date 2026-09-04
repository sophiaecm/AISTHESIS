"""Local scene description for Fifth Layer Engine v0.13."""


def describe_scene(scene_graph):
    """
    Convert a SceneGraph into a simple local natural-language description.
    """

    sentences = []

    for node in scene_graph.nodes:
        sentence = (
            f"A {node.class_name} is in the "
            f"{node.horizontal_position} part of the scene "
            f"and vertically in the {node.vertical_position}."
        )

        sentences.append(sentence)

    for relation in scene_graph.relations:
        source = scene_graph.nodes[relation.source_id]
        target = scene_graph.nodes[relation.target_id]

        if relation.relation == "left_of":
            sentences.append(
                f"The {source.class_name} is left of the "
                f"{target.class_name}."
            )

        elif relation.relation == "right_of":
            sentences.append(
                f"The {source.class_name} is right of the "
                f"{target.class_name}."
            )

        elif relation.relation == "near":
            sentences.append(
                f"The {source.class_name} is near the "
                f"{target.class_name}."
            )

        elif relation.relation == "far":
            sentences.append(
                f"The {source.class_name} is far from the "
                f"{target.class_name}."
            )

        elif relation.relation == "overlapping":
            sentences.append(
                f"The {source.class_name} overlaps the "
                f"{target.class_name}."
            )

        elif relation.relation == "separate":
            sentences.append(
                f"The {source.class_name} is separate from the "
                f"{target.class_name}."
            )

    return " ".join(sentences)