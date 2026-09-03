"""Probabilistic reasoning for Fifth Layer Engine v0.6."""

from fifth_layer.world_state import WorldState
from fifth_layer.expected_consequences import ExpectedConsequences
from fifth_layer.latent_state import LatentState
from fifth_layer.future_state import FutureState
from fifth_layer.reasoners.base import BaseReasoner


class ProbabilisticReasoner(BaseReasoner):
    """A simple uncertainty-aware reasoner using explicit probabilities."""

    def infer_expected_consequences(
        self,
        world_state: WorldState,
    ) -> ExpectedConsequences:
        data = world_state.data
        predictions = {}

        hidden_actor_probability = 0.10

        if data.get("occlusion") is True:
            hidden_actor_probability += 0.25

        if data.get("object_motion") == "toward_occlusion":
            hidden_actor_probability += 0.20

        if data.get("object_type") == "ball":
            hidden_actor_probability += 0.15

        if data.get("trajectory_continues") is True:
            hidden_actor_probability += 0.10

        hidden_actor_probability = min(
            hidden_actor_probability,
            0.95,
        )

        no_hidden_actor_probability = 1.0 - hidden_actor_probability

        predictions["hidden_actor_probability"] = round(
            hidden_actor_probability,
            3,
        )

        predictions["no_hidden_actor_probability"] = round(
            no_hidden_actor_probability,
            3,
        )

        predictions["uncertainty"] = round(
            1.0 - abs(hidden_actor_probability - 0.5) * 2.0,
            3,
        )

        return ExpectedConsequences(predictions=predictions)

    def infer_latent_state(
        self,
        world_state: WorldState,
        expected_consequences: ExpectedConsequences,
    ) -> LatentState:
        features = dict(world_state.data)
        features.update(expected_consequences.predictions)

        probability = features.get(
            "hidden_actor_probability",
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