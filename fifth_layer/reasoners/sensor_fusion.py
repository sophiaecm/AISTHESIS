"""Sensor fusion reasoning for Fifth Layer Engine."""

from fifth_layer.reasoners.occlusion_context import occlusion_context


from fifth_layer.world_state import WorldState
from fifth_layer.expected_consequences import ExpectedConsequences
from fifth_layer.latent_state import LatentState
from fifth_layer.future_state import FutureState
from fifth_layer.reasoners.base import BaseReasoner


class SensorFusionReasoner(BaseReasoner):
    """
    Fuse evidence conservatively.

    Important:
    - Generic occlusion does NOT imply a hidden actor.
    - Frame truncation alone does NOT imply a hidden actor.
    - Hidden actor inference requires actor-specific overlap
      or another explicit actor-related cue.
    """

    def infer_expected_consequences(
        self,
        world_state: WorldState,
    ) -> ExpectedConsequences:

        data = world_state.data

        predictions = {}

        hidden_actor_evidence = []
        generic_evidence = []

        # --------------------------------------------------
        # 1. Actor-specific visual evidence
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
            hidden_actor_evidence.append(
                {
                    "source": "vision",
                    "confidence": vision_confidence,
                }
            )

        # --------------------------------------------------
        # 2. Actor-specific motion evidence
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
            hidden_actor_evidence.append(
                {
                    "source": "motion",
                    "confidence": motion_confidence,
                }
            )

        # --------------------------------------------------
        # 3. Actor-specific physics evidence
        # --------------------------------------------------

        physics_confidence = float(
            data.get(
                "physics_confidence",
                0.0,
            )
        )

        physics_hidden_interaction = bool(
            data.get(
                "physics_hidden_interaction_possible",
                False,
            )
        )

        if physics_hidden_interaction:
            hidden_actor_evidence.append(
                {
                    "source": "physics",
                    "confidence": physics_confidence,
                }
            )

        # --------------------------------------------------
        # 4. Occlusion evidence
        # --------------------------------------------------

        occlusion_evidence = data.get(
            "occlusion_evidence",
            [],
        )

        generic_occlusion_scores = []
        actor_occlusion_scores = []

        actor_terms = {
            "person",
            "pedestrian",
            "human",
            "child",
            "cyclist",
            "bicycle",
            "motorcycle",
            "rider",
        }

        for item in occlusion_evidence:

            probability = 0.0

            frame_truncated = bool(
                item.get(
                    "frame_truncated",
                    False,
                )
            )

            has_overlap = bool(
                item.get(
                    "has_overlap_evidence",
                    False,
                )
            )

            overlaps = item.get(
                "overlapping_objects",
                [],
            )

            # ----------------------------------------------
            # Generic occlusion score
            # ----------------------------------------------

            if frame_truncated:
                probability += 0.35

            if has_overlap:
                probability += 0.35

            strongest_overlap = 0.0

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
                generic_occlusion_scores.append(
                    probability
                )

            # ----------------------------------------------
            # Actor-specific occlusion score
            #
            # IMPORTANT:
            # Frame truncation alone is not enough.
            # We require real overlap evidence.
            # ----------------------------------------------

            class_name = str(
                item.get(
                    "class_name",
                    "",
                )
            ).lower()

            if (
                class_name in actor_terms
                and has_overlap
                and strongest_overlap > 0.0
            ):
                actor_probability = (
                    0.35
                    + min(
                        strongest_overlap,
                        0.45,
                    )
                )

                actor_probability = min(
                    actor_probability,
                    0.90,
                )

                actor_occlusion_scores.append(
                    actor_probability
                )

        # --------------------------------------------------
        # Generic occlusion output
        # --------------------------------------------------

        context = occlusion_context(data)
        if context is not None:
            hypotheses, latent_objects = context
            # Same visual evidence: merge by maximum, do not count another sensor.
            generic_occlusion_scores.extend(
                item["occlusion_probability"] for item in hypotheses
                if item["occlusion_probability"] > 0
            )
            predictions["possible_occluded_object_count"] = len(latent_objects)

        if generic_occlusion_scores:

            strongest_generic = max(
                generic_occlusion_scores
            )

            generic_evidence.append(
                {
                    "source": "occlusion",
                    "confidence": strongest_generic,
                }
            )

            predictions[
                "generic_occlusion_probability"
            ] = round(
                strongest_generic,
                3,
            )

        else:

            predictions[
                "generic_occlusion_probability"
            ] = 0.0

        # --------------------------------------------------
        # Visible actor-overlap diagnostic
        # --------------------------------------------------
        #
        # A visible actor overlapping another visible object
        # is NOT evidence that an additional hidden actor exists.
        # Keep this only as a visibility/occlusion diagnostic.
        # --------------------------------------------------

        if actor_occlusion_scores:

            strongest_actor_occlusion = max(
                actor_occlusion_scores
            )

            predictions[
                "visible_actor_occlusion_probability"
            ] = round(
                strongest_actor_occlusion,
                3,
            )

            generic_evidence.append(
                {
                    "source": "visible_actor_occlusion",
                    "confidence": strongest_actor_occlusion,
                }
            )

        else:

            predictions[
                "visible_actor_occlusion_probability"
            ] = 0.0

        # --------------------------------------------------
        # 5. Semantic evidence
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
        # 6. Hidden actor fusion
        # --------------------------------------------------

        confidence_values = [
            item["confidence"]
            for item in hidden_actor_evidence
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
        # 7. Uncertainty
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
        ] = len(
            hidden_actor_evidence
        )

        predictions[
            "evidence_sources"
        ] = [
            item["source"]
            for item in hidden_actor_evidence
        ]

        predictions[
            "generic_evidence_sources"
        ] = [
            item["source"]
            for item in generic_evidence
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

        active_sources = int(
            features.get(
                "active_evidence_sources",
                0,
            )
        )

        if active_sources == 0:

            features[
                "latent_hypothesis"
            ] = "insufficient_hidden_actor_evidence"

        elif probability >= 0.70:

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

        hypothesis = (
            latent_state.features.get(
                "latent_hypothesis"
            )
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
            ] = "hidden_actor_may_emerge"

            future_data[
                "risk_level"
            ] = "high"

        elif hypothesis == "hidden_actor_possible":

            future_data[
                "predicted_event"
            ] = "hidden_actor_may_emerge"

            future_data[
                "risk_level"
            ] = "medium"

        elif hypothesis == "hidden_actor_unlikely":

            future_data[
                "predicted_event"
            ] = "no_actor_expected"

            future_data[
                "risk_level"
            ] = "low"

        else:

            future_data[
                "predicted_event"
            ] = "indeterminate"

            future_data[
                "risk_level"
            ] = "UNKNOWN"

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
