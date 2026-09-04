"""Occlusion reasoning for Fifth Layer Engine v0.18."""

from fifth_layer.expected_consequences import ExpectedConsequences
from fifth_layer.future_state import FutureState
from fifth_layer.latent_state import LatentState
from fifth_layer.reasoners.base import BaseReasoner
from fifth_layer.world_state import WorldState


class OcclusionReasoner(BaseReasoner):
    """Infer uncertainty-aware occlusion hypotheses from visual evidence."""

    def infer_expected_consequences(
        self,
        world_state: WorldState,
    ) -> ExpectedConsequences:
        evidence = world_state.data.get(
            "occlusion_evidence",
            [],
        )

        predictions = {
            "occlusion_hypotheses": []
        }

        for item in evidence:
            probability = 0.0

            if item.get("frame_truncated"):
                probability += 0.35

            if item.get("has_overlap_evidence"):
                probability += 0.35

            overlapping_objects = item.get(
                "overlapping_objects",
                [],
            )

            if overlapping_objects:
                strongest_overlap = max(
                    overlap.get(
                        "overlap_ratio",
                        0.0,
                    )
                    for overlap in overlapping_objects
                )

                probability += min(
                    strongest_overlap,
                    0.25,
                )

            probability = min(
                probability,
                0.95,
            )

            if probability >= 0.70:
                hypothesis = "occlusion_likely"

            elif probability >= 0.30:
                hypothesis = "occlusion_possible"

            else:
                hypothesis = "occlusion_unlikely"

            predictions[
                "occlusion_hypotheses"
            ].append(
                {
                    "object_id": item.get(
                        "object_id"
                    ),
                    "class_name": item.get(
                        "class_name"
                    ),
                    "occlusion_probability": round(
                        probability,
                        3,
                    ),
                    "hypothesis": hypothesis,
                }
            )

        return ExpectedConsequences(
            predictions=predictions
        )

    def infer_latent_state(
        self,
        world_state: WorldState,
        expected_consequences: ExpectedConsequences,
    ) -> LatentState:
        features = dict(
            world_state.data
        )

        features.update(
            expected_consequences.predictions
        )

        hypotheses = features.get(
            "occlusion_hypotheses",
            [],
        )

        possible_objects = []

        for hypothesis in hypotheses:
            if hypothesis.get(
                "hypothesis"
            ) in {
                "occlusion_possible",
                "occlusion_likely",
            }:
                possible_objects.append(
                    hypothesis
                )

        features[
            "latent_occlusion_state"
        ] = {
            "possible_occluded_objects": possible_objects,
            "count": len(possible_objects),
        }

        return LatentState(
            features=features
        )

    def infer_future_state(
        self,
        latent_state: LatentState,
    ) -> FutureState:
        future_data = dict(
            latent_state.features
        )

        latent_occlusion_state = (
            latent_state.features.get(
                "latent_occlusion_state",
                {},
            )
        )

        count = latent_occlusion_state.get(
            "count",
            0,
        )

        if count > 0:
            future_data[
                "predicted_occlusion_event"
            ] = "partially_hidden_content_may_become_visible"

        else:
            future_data[
                "predicted_occlusion_event"
            ] = "no_occlusion_change_expected"

        return FutureState(
            horizon=1.0,
            data=future_data,
        )