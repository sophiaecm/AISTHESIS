from time import time

from fifth_layer.world_state import WorldState
from fifth_layer.perception.occlusion import extract_occlusion_evidence
from fifth_layer.perception.spatial import (
    distance_relation,
    horizontal_relation,
    overlap_relation,
)
from fifth_layer.perception.semantic_evidence import (
    extract_semantic_evidence,
)


class PerceptionFusion:
    """Combine detector and scene perception into one WorldState."""

    def _build_scene_relations(
        self,
        detections,
        image_width,
        image_height,
    ):
        relations = []

        for i, detection_a in enumerate(detections):
            for j, detection_b in enumerate(detections):
                if i >= j:
                    continue

                class_a = detection_a.get(
                    "class_name",
                    "unknown",
                )

                class_b = detection_b.get(
                    "class_name",
                    "unknown",
                )

                overlap = overlap_relation(
                    detection_a,
                    detection_b,
                )

                distance = distance_relation(
                    detection_a,
                    detection_b,
                    image_width,
                    image_height,
                )

                horizontal = horizontal_relation(
                    detection_a,
                    detection_b,
                )

                relations.append(
                    {
                        "first_object_id": i,
                        "second_object_id": j,
                        "first_class": class_a,
                        "second_class": class_b,
                        "relation": overlap,
                    }
                )

                relations.append(
                    {
                        "first_object_id": i,
                        "second_object_id": j,
                        "first_class": class_a,
                        "second_class": class_b,
                        "relation": distance,
                    }
                )

                relations.append(
                    {
                        "first_object_id": i,
                        "second_object_id": j,
                        "first_class": class_a,
                        "second_class": class_b,
                        "relation": horizontal,
                    }
                )

        return relations

    def fuse(
        self,
        yolo_state: WorldState,
        smolvlm_state: WorldState,
    ) -> WorldState:

        yolo_data = yolo_state.data
        smolvlm_data = smolvlm_state.data

        detections = yolo_data.get(
            "detections",
            [],
        )

        image_width = yolo_data.get(
            "image_width",
            0,
        )

        image_height = yolo_data.get(
            "image_height",
            0,
        )

        scene_description = smolvlm_data.get(
            "scene_description",
            "",
        )

        # --------------------------------------------------
        # Structured semantic evidence from SmolVLM
        # --------------------------------------------------

        semantic_evidence = extract_semantic_evidence(
            scene_description
        )

        # --------------------------------------------------
        # Spatial and occlusion evidence from detections
        # --------------------------------------------------

        if (
            detections
            and image_width
            and image_height
        ):
            scene_relations = (
                self._build_scene_relations(
                    detections,
                    image_width,
                    image_height,
                )
            )

            occlusion_evidence = (
                extract_occlusion_evidence(
                    detections=detections,
                    image_width=image_width,
                    image_height=image_height,
                )
            )

        else:
            scene_relations = []
            occlusion_evidence = []

        # --------------------------------------------------
        # Unified WorldState
        # --------------------------------------------------

        return WorldState(
            timestamp=time(),
            data={
                "source_type": yolo_data.get(
                    "source_type",
                    "image",
                ),

                "source_path": yolo_data.get(
                    "source_path",
                ),

                "image_width": image_width,

                "image_height": image_height,

                "detections": detections,

                "detection_count": yolo_data.get(
                    "detection_count",
                    len(detections),
                ),

                "scene_description": scene_description,

                "semantic_evidence": semantic_evidence,

                "scene_relations": scene_relations,

                "occlusion_evidence": occlusion_evidence,

                "perception_sources": {
                    "detector": yolo_data.get(
                        "model",
                    ),

                    "scene_model": smolvlm_data.get(
                        "model",
                    ),
                },
            },
        )