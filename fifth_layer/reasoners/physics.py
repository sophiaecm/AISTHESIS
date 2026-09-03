"""Physics-based reasoning for Fifth Layer Engine v0.5."""

from fifth_layer.world_state import WorldState
from fifth_layer.expected_consequences import ExpectedConsequences
from fifth_layer.latent_state import LatentState
from fifth_layer.future_state import FutureState
from fifth_layer.reasoners.base import BaseReasoner


class PhysicsReasoner(BaseReasoner):
    """A simple deterministic 2D physics reasoner."""

    def infer_expected_consequences(
        self,
        world_state: WorldState,
    ) -> ExpectedConsequences:
        data = world_state.data
        predictions = {}

        position = data.get("position")
        velocity = data.get("velocity")
        dt = data.get("dt", 1.0)

        if position is not None and velocity is not None:
            x, y = position
            vx, vy = velocity

            next_position = (
                x + vx * dt,
                y + vy * dt,
            )

            predictions["expected_next_position"] = next_position

            occlusion_zone = data.get("occlusion_zone")

            if occlusion_zone is not None:
                xmin, ymin, xmax, ymax = occlusion_zone
                nx, ny = next_position

                inside_occlusion = (
                    xmin <= nx <= xmax
                    and ymin <= ny <= ymax
                )

                predictions["enters_occlusion_zone"] = inside_occlusion

                if inside_occlusion:
                    predictions["hidden_interaction_possible"] = True

        return ExpectedConsequences(predictions=predictions)

    def infer_latent_state(
        self,
        world_state: WorldState,
        expected_consequences: ExpectedConsequences,
    ) -> LatentState:
        features = dict(world_state.data)
        features.update(expected_consequences.predictions)

        if features.get("hidden_interaction_possible"):
            features["latent_physical_risk"] = (
                "possible_occluded_interaction"
            )

        return LatentState(features=features)

    def infer_future_state(
        self,
        latent_state: LatentState,
    ) -> FutureState:
        future_data = dict(latent_state.features)

        if (
            latent_state.features.get("latent_physical_risk")
            == "possible_occluded_interaction"
        ):
            future_data["predicted_event"] = (
                "object_may_interact_inside_occlusion"
            )

        return FutureState(
            horizon=1.0,
            data=future_data,
        )