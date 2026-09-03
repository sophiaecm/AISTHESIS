"""PlaceholderReasoner: a trivial reasoner used to validate the architecture.

This exists only to prove that FifthLayerEngine can be driven by a
pluggable reasoner. All logic below is PLACEHOLDER LOGIC ONLY - it does
not perform real inference, prediction, or learning.
"""

from fifth_layer.reasoners.base import BaseReasoner
from fifth_layer.world_state import WorldState
from fifth_layer.expected_consequences import ExpectedConsequences
from fifth_layer.latent_state import LatentState
from fifth_layer.future_state import FutureState


class PlaceholderReasoner(BaseReasoner):
    """A minimal reasoner with no real reasoning - for architecture testing.

    This is the same placeholder logic used in the v0.2 pipeline, now
    moved out of the engine so it can be swapped for other reasoners.
    """

    def infer_expected_consequences(self, world_state: WorldState) -> ExpectedConsequences:
        """PLACEHOLDER LOGIC: echo the observed data as the expectation."""
        return ExpectedConsequences(predictions=dict(world_state.data))

    def infer_latent_state(
        self,
        world_state: WorldState,
        expected_consequences: ExpectedConsequences,
    ) -> LatentState:
        """PLACEHOLDER LOGIC: merge observed data and expected consequences."""
        features = dict(world_state.data)
        features.update(expected_consequences.predictions)
        return LatentState(features=features)

    def infer_future_state(self, latent_state: LatentState) -> FutureState:
        """PLACEHOLDER LOGIC: carry latent features forward unchanged."""
        return FutureState(horizon=1.0, data=dict(latent_state.features))
