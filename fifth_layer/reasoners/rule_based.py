"""Rule-based reasoning for Fifth Layer Engine v0.4."""

from fifth_layer.world_state import WorldState
from fifth_layer.expected_consequences import ExpectedConsequences
from fifth_layer.latent_state import LatentState
from fifth_layer.future_state import FutureState
from fifth_layer.reasoners.base import BaseReasoner


class RuleBasedReasoner(BaseReasoner):
    """A simple deterministic reasoner based on explicit rules."""

    def infer_expected_consequences(
        self,
        world_state: WorldState,
    ) -> ExpectedConsequences:
        predictions = {}

        data = world_state.data

        if data.get("object_motion") == "toward_occlusion":
            predictions["possible_hidden_interaction"] = True

        if data.get("object_type") == "ball":
            predictions["trajectory_continues"] = True

        if data.get("occlusion") is True:
            predictions["hidden_actor_possible"] = True

        return ExpectedConsequences(predictions=predictions)

    def infer_latent_state(
        self,
        world_state: WorldState,
        expected_consequences: ExpectedConsequences,
    ) -> LatentState:
        features = dict(world_state.data)
        features.update(expected_consequences.predictions)

        if (
            features.get("hidden_actor_possible")
            and features.get("trajectory_continues")
        ):
            features["latent_risk"] = "possible_actor_emergence"

        return LatentState(features=features)

    def infer_future_state(
        self,
        latent_state: LatentState,
    ) -> FutureState:
        future_data = dict(latent_state.features)

        if latent_state.features.get("latent_risk") == "possible_actor_emergence":
            future_data["predicted_event"] = "actor_may_emerge"

        return FutureState(
            horizon=1.0,
            data=future_data,
        )