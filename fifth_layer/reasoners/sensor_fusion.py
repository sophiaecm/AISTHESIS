"""Sensor fusion reasoning for Fifth Layer Engine v0.7."""

from fifth_layer.world_state import WorldState
from fifth_layer.expected_consequences import ExpectedConsequences
from fifth_layer.latent_state import LatentState
from fifth_layer.future_state import FutureState
from fifth_layer.reasoners.base import BaseReasoner


class SensorFusionReasoner(BaseReasoner):
    """Fuse multiple sensor observations using confidence-weighted evidence."""

    def infer_expected_consequences(
        self,
        world_state: WorldState,
    ) -> ExpectedConsequences:
        data = world_state.data
        predictions = {}

        vision_confidence = float(data.get("vision_confidence", 0.0))
        motion_confidence = float(data.get("motion_confidence", 0.0))
        physics_confidence = float(data.get("physics_confidence", 0.0))

        vision_hidden_actor = bool(
            data.get("vision_hidden_actor_possible", False)
        )
        motion_toward_occlusion = bool(
            data.get("motion_toward_occlusion", False)
        )
        physics_interaction = bool(
            data.get("physics_hidden_interaction_possible", False)
        )

        evidence = []

        if vision_hidden_actor:
            evidence.append(vision_confidence)

        if motion_toward_occlusion:
            evidence.append(motion_confidence)

        if physics_interaction:
            evidence.append(physics_confidence)

        if evidence:
            fused_probability = sum(evidence) / len(evidence)
        else:
            fused_probability = 0.0

        fused_probability = max(
            0.0,
            min(fused_probability, 1.0),
        )

        predictions["fused_hidden_actor_probability"] = round(
            fused_probability,
            3,
        )

        predictions["fused_uncertainty"] = round(
            1.0 - abs(fused_probability - 0.5) * 2.0,
            3,
        )

        predictions["active_evidence_sources"] = len(evidence)

        return ExpectedConsequences(predictions=predictions)

    def infer_latent_state(
        self,
        world_state: WorldState,
        expected_consequences: ExpectedConsequences,
    ) -> LatentState:
        features = dict(world_state.data)
        features.update(expected_consequences.predictions)

        probability = features.get(
            "fused_hidden_actor_probability",
            0.0,
        )

        if probability >= 0.70:
            features["latent_hypothesis"] = "hidden_actor_likely"
        elif probability >= 0.40:
            features["latent_hypothesis"] = "hidden_actor_possible"
        else:
            features["latent_hypothesis"] = "hidden_actor_unlikely"

        return LatentState(features=features)

    def infer_future_state(
        self,
        latent_state: LatentState,
    ) -> FutureState:
        future_data = dict(latent_state.features)

        hypothesis = latent_state.features.get("latent_hypothesis")

        if hypothesis == "hidden_actor_likely":
            future_data["predicted_event"] = "actor_may_emerge"
            future_data["risk_level"] = "high"

        elif hypothesis == "hidden_actor_possible":
            future_data["predicted_event"] = "actor_may_emerge"
            future_data["risk_level"] = "medium"

        else:
            future_data["predicted_event"] = "no_actor_expected"
            future_data["risk_level"] = "low"

        return FutureState(
            horizon=1.0,
            data=future_data,
        )