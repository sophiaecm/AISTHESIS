"""Composite reasoner for Fifth Layer Engine v0.15."""

from fifth_layer.expected_consequences import ExpectedConsequences
from fifth_layer.future_state import FutureState
from fifth_layer.latent_state import LatentState
from fifth_layer.reasoners.base import BaseReasoner
from fifth_layer.world_state import WorldState


class CompositeReasoner(BaseReasoner):
    """Run multiple reasoners and combine their outputs by namespace."""

    def __init__(self, reasoners):
        self.reasoners = reasoners

    def infer_expected_consequences(
        self,
        world_state: WorldState,
    ) -> ExpectedConsequences:
        combined_predictions = {}

        for name, reasoner in self.reasoners.items():
            result = reasoner.infer_expected_consequences(
                world_state
            )

            combined_predictions[name] = (
                result.predictions
            )

        return ExpectedConsequences(
            predictions=combined_predictions
        )

    def infer_latent_state(
        self,
        world_state: WorldState,
        expected_consequences: ExpectedConsequences,
    ) -> LatentState:
        combined_features = {
            "world_state": dict(world_state.data),
            "reasoners": {},
        }

        for name, reasoner in self.reasoners.items():
            predictions = (
                expected_consequences.predictions.get(
                    name,
                    {},
                )
            )

            local_expected = ExpectedConsequences(
                predictions=predictions
            )

            latent_state = reasoner.infer_latent_state(
                world_state,
                local_expected,
            )

            combined_features[
                "reasoners"
            ][name] = latent_state.features

        return LatentState(
            features=combined_features
        )

    def infer_future_state(
        self,
        latent_state: LatentState,
    ) -> FutureState:
        combined_future = {
            "reasoners": {}
        }

        reasoner_features = (
            latent_state.features.get(
                "reasoners",
                {},
            )
        )

        for name, reasoner in self.reasoners.items():
            features = reasoner_features.get(
                name,
                {},
            )

            local_latent = LatentState(
                features=features
            )

            future_state = (
                reasoner.infer_future_state(
                    local_latent
                )
            )

            combined_future[
                "reasoners"
            ][name] = future_state.data

        return FutureState(
            horizon=1.0,
            data=combined_future,
        )