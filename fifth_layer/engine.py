"""FifthLayerEngine: model-independent orchestration for the Fifth Layer.

As of v0.3, transformation/inference logic is delegated to a pluggable
reasoner. The core engine itself does not depend on any language model,
machine-learning model, external API, or network service.
"""

from typing import Any, Dict, Optional

from fifth_layer.world_state import WorldState
from fifth_layer.future_state import FutureState
from fifth_layer.prediction_error import PredictionError
from fifth_layer.reasoners.base import BaseReasoner
from fifth_layer.reasoners.placeholder import PlaceholderReasoner


class FifthLayerEngine:
    """Orchestrate the Fifth Layer pipeline using a replaceable reasoner."""

    def __init__(self, reasoner: Optional[BaseReasoner] = None) -> None:
        """Create the engine with an optional reasoning strategy."""
        self.reasoner = (
            reasoner if reasoner is not None else PlaceholderReasoner()
        )
        self.last_future_state: Optional[FutureState] = None

    def step(self, world_state: WorldState) -> Dict[str, Any]:
        """Run one complete reasoning pass from an observed WorldState."""

        expected_consequences = (
            self.reasoner.infer_expected_consequences(world_state)
        )

        latent_state = self.reasoner.infer_latent_state(
            world_state,
            expected_consequences,
        )

        future_state = self.reasoner.infer_future_state(latent_state)

        self.last_future_state = future_state

        return {
            "world_state": world_state,
            "expected_consequences": expected_consequences,
            "latent_state": latent_state,
            "future_state": future_state,
        }

    def compare(self, observed: WorldState) -> PredictionError:
        """Compare a later observation with the most recent FutureState."""

        if self.last_future_state is None:
            raise ValueError(
                "No FutureState to compare against. Call step() first."
            )

        details: Dict[str, Any] = {}

        predicted = self.last_future_state.data
        actual = observed.data

        for key in set(predicted) | set(actual):
            predicted_value = predicted.get(key)
            actual_value = actual.get(key)

            if predicted_value != actual_value:
                details[key] = {
                    "expected": predicted_value,
                    "actual": actual_value,
                }

        return PredictionError(details=details)