"""Sensor fusion reasoning for Fifth Layer Engine."""

from fifth_layer.world_state import WorldState
from fifth_layer.expected_consequences import ExpectedConsequences
from fifth_layer.latent_state import LatentState
from fifth_layer.future_state import FutureState
from fifth_layer.reasoners.base import BaseReasoner


class SensorFusionReasoner(BaseReasoner):
    """
    Fuse visual, motion, physics, and occlusion evidence.

    The reasoner does not assume that missing evidence means absence.
    It only increases hidden-state probability when observable evidence
    supports a latent hypothesis.
    """

    def infer_expected_consequences(
        self,
        world_state: WorldState,
    ) -> ExpectedConsequences:

        data = world_state.data

        predictions = {}

        evidence = []

        # --------------------------------------------------
        # Existing visual evidence
        # --------------------------------------------------

        vision_confidence = float(
            data.get(
                "vision_confidence",
                0.0,
            )
        )

        vision_hidden_actor = bool(
            data.get(
                "vision_hidden_actor_possible",
                False,
            )
        )

        if vision_hidden_actor:
            evidence.append(
                {
                    "source": "vision",
                    "confidence": vision_confidence,
                }
            )

        # --------------------------------------------------
        # Motion evidence
        # --------------------------------------------------

        motion_confidence = float(
            data.get(
                "motion_confidence",
                0.0,
            )
        )

        motion_toward_occlusion = bool(
            data.get(
                "motion_toward_occlusion",
                False,
            )
        )

        if motion_toward_occlusion:
            evidence.append(
                {
                    "source": "motion",
                    "confidence": motion_confidence,
                }
            )

        # --------------------------------------------------
        # Physics evidence
        # --------------------------------------------------

        physics_confidence = float(
            data.get(
                "physics_confidence",
                0.0,
            )
        )

        physics_interaction = bool(
            data.get(
                "physics_hidden_interaction_possible",
                False,
            )
        )

        if physics_interaction:
            evidence.append(
                {
                    "source": "physics",
                    "confidence": physics_confidence,
                }
            )

        # --------------------------------------------------
        # Occlusion evidence
        # --------------------------------------------------

        occlusion_evidence = data.get(
            "occlusion_evidence",
            [],
        )

        occlusion_scores = []

        for item in occlusion_evidence:

            probability = 0.0

            if item.get(
                "frame_truncated",
                False,
            ):
                probability += 0.35

            if item.get(
                "has_overlap_evidence",
                False,
            ):
                probability += 0.35

            overlaps = item.get(
                "overlapping_objects",
                [],
            )

            if overlaps:

                strongest_overlap = max(
                    (
                        float(
                            overlap.get(
                                "overlap_ratio",
                                0.0,
                            )
                        )
                        for overlap in overlaps
                    ),
                    default=0.0,
                )

                probability += min(
                    strongest_overlap,
                    0.25,
                )

            probability = min(
                probability,
                0.95,
            )

            if probability > 0:
                occlusion_scores.append(
                    probability
                )

        if occlusion_scores:

            strongest_occlusion = max(
                occlusion_scores
            )

            evidence.append(
                {
                    "source": "occlusion",
                    "confidence": strongest_occlusion,
                }
            )

        # --------------------------------------------------
        # Semantic evidence
        # --------------------------------------------------

        semantic_evidence = data.get(
            "semantic_evidence",
            {},
        )

        mentioned_objects = semantic_evidence.get(
            "mentioned_objects",
            [],
        )

        mentioned_actions = semantic_evidence.get(
            "mentioned_actions",
            [],
        )

        if mentioned_objects:

            predictions[
                "semantic_object_evidence"
            ] = mentioned_objects

        if mentioned_actions:

            predictions[
                "semantic_action_evidence"
            ] = mentioned_actions

        # --------------------------------------------------
        # Fuse evidence
        # --------------------------------------------------

        confidence_values = [
            item["confidence"]
            for item in evidence
            if item["confidence"] > 0
        ]

        if confidence_values:

            fused_probability = (
                sum(confidence_values)
                / len(confidence_values)
            )

        else:

            fused_probability = 0.0

        fused_probability = max(
            0.0,
            min(
                fused_probability,
                1.0,
            ),
        )

        predictions[
            "fused_hidden_actor_probability"
        ] = round(
            fused_probability,
            3,
        )

        # --------------------------------------------------
        # Uncertainty
        # --------------------------------------------------

        if confidence_values:

            uncertainty = (
                1.0
                - abs(
                    fused_probability
                    - 0.5
                )
                * 2.0
            )

        else:

            uncertainty = 1.0

        predictions[
            "fused_uncertainty"
        ] = round(
            max(
                0.0,
                min(
                    uncertainty,
                    1.0,
                ),
            ),
            3,
        )

        predictions[
            "active_evidence_sources"
        ] = len(evidence)

        predictions[
            "evidence_sources"
        ] = [
            item["source"]
            for item in evidence
        ]

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

        probability = float(
            features.get(
                "fused_hidden_actor_probability",
                0.0,
            )
        )

        if probability >= 0.70:

            features[
                "latent_hypothesis"
            ] = "hidden_actor_likely"

        elif probability >= 0.40:

            features[
                "latent_hypothesis"
            ] = "hidden_actor_possible"

        else:

            features[
                "latent_hypothesis"
            ] = "hidden_actor_unlikely"

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

        hypothesis = latent_state.features.get(
            "latent_hypothesis"
        )

        probability = float(
            latent_state.features.get(
                "fused_hidden_actor_probability",
                0.0,
            )
        )

        if hypothesis == "hidden_actor_likely":

            future_data[
                "predicted_event"
            ] = "actor_may_emerge"

            future_data[
                "risk_level"
            ] = "high"

        elif hypothesis == "hidden_actor_possible":

            future_data[
                "predicted_event"
            ] = "actor_may_emerge"

            future_data[
                "risk_level"
            ] = "medium"

        else:

            future_data[
                "predicted_event"
            ] = "no_actor_expected"

            future_data[
                "risk_level"
            ] = "low"

        future_data[
            "prediction_confidence"
        ] = round(
            probability,
            3,
        )

        return FutureState(
            horizon=1.0,
            data=future_data,
        )