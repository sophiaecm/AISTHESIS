"""Scene graph representation for Fifth Layer Engine v0.12."""

from dataclasses import dataclass
from typing import List

from fifth_layer.perception.spatial import (
    horizontal_position,
    vertical_position,
    horizontal_relation,
    distance_relation,
    overlap_relation,
)


@dataclass
class SceneNode:
    """A detected object represented as a node."""

    node_id: int
    class_name: str
    confidence: float
    box: list
    horizontal_position: str
    vertical_position: str


@dataclass
class SceneRelation:
    """A relation between two scene nodes."""

    source_id: int
    relation: str
    target_id: int


@dataclass
class SceneGraph:
    """Structured representation of objects and relations."""

    nodes: List[SceneNode]
    relations: List[SceneRelation]


def build_scene_graph(
    detections,
    image_width,
    image_height,
):
    """
    Build a simple scene graph from object detections.
    """

    nodes = []
    relations = []

    for index, detection in enumerate(detections):
        node = SceneNode(
            node_id=index,
            class_name=detection["class_name"],
            confidence=detection["confidence"],
            box=detection["box"],
            horizontal_position=horizontal_position(
                detection["box"],
                image_width,
            ),
            vertical_position=vertical_position(
                detection["box"],
                image_height,
            ),
        )

        nodes.append(node)

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            first = nodes[i]
            second = nodes[j]

            relations.append(
                SceneRelation(
                    source_id=first.node_id,
                    relation=horizontal_relation(
                        first.box,
                        second.box,
                    ),
                    target_id=second.node_id,
                )
            )

            relations.append(
                SceneRelation(
                    source_id=first.node_id,
                    relation=distance_relation(
                        first.box,
                        second.box,
                        image_width,
                        image_height,
                    ),
                    target_id=second.node_id,
                )
            )

            relations.append(
                SceneRelation(
                    source_id=first.node_id,
                    relation=overlap_relation(
                        first.box,
                        second.box,
                    ),
                    target_id=second.node_id,
                )
            )

    return SceneGraph(
        nodes=nodes,
        relations=relations,
    )