"""FifthLayerEngine: connects the five core concepts into one pipeline.

This is a v0.2 scaffold. All transformation logic below is PLACEHOLDER
LOGIC ONLY - it does not perform real inference, prediction, or learning.
"""

from typing import Any, Dict, Optional

from fifth_layer.world_state import WorldState
from fifth_layer.expected_consequences import ExpectedConsequences
from fifth_layer.latent_state import LatentState
from fifth_layer.future_state import FutureState
from fifth_layer.prediction_error import PredictionError


class FifthLayerEngine:
    """A minimal, linear pipeline connecting the five core concepts.

    The pipeline runs in two parts:

    1. ``step`` takes a WorldState and produces ExpectedConsequences,
       a LatentState, and a FutureState (a prediction of what comes next).
    2. ``compare`` takes a later, actually observed WorldState and compares
       it against the previously produced FutureState to produce a
       PredictionError.
    """

    def __init__(self) -> None:
        """Create an engine with no prior prediction yet."""
        self.last_future_state: Optional[FutureState] = None

    def step(self, world_state: WorldState) -> Dict[str, Any]:
        """Run one full pipeline pass starting from a WorldState.

        Returns a dict with every intermediate object so the pipeline
        can be inspected: "world_state", "expected_consequences",
        "latent_state", and "future_state".
        """
        expected_consequences = self._predict_consequences(world_state)
        latent_state = self._encode_latent_state(world_state, expected_consequences)
        future_state = self._project_future_state(latent_state)

        self.last_future_state = future_state

        return {
            "world_state": world_state,
            "expected_consequences": expected_consequences,
            "latent_state": latent_state,
            "future_state": future_state,
        }

    def compare(self, observed: WorldState) -> PredictionError:
        """Compare a later observed WorldState with the last FutureState.

        Raises a ValueError if no prediction has been made yet (i.e. if
        ``step`` has not been called before ``compare``).
        """
        if self.last_future_state is None:
            raise ValueError("No FutureState to compare against. Call step() first.")

        # PLACEHOLDER LOGIC: compare matching keys directly and record any
        # mismatches, including keys missing from either side.
        details: Dict[str, Any] = {}
        predicted = self.last_future_state.data
        actual = observed.data

        for key in set(predicted) | set(actual):
            predicted_value = predicted.get(key)
            actual_value = actual.get(key)
            if predicted_value != actual_value:
                details[key] = {"expected": predicted_value, "actual": actual_value}

        return PredictionError(details=details)

    def _predict_consequences(self, world_state: WorldState) -> ExpectedConsequences:
        """PLACEHOLDER LOGIC: echo the observed data as the expectation."""
        return ExpectedConsequences(predictions=dict(world_state.data))

    def _encode_latent_state(
        self,
        world_state: WorldState,
        expected_consequences: ExpectedConsequences,
    ) -> LatentState:
        """PLACEHOLDER LOGIC: merge observed data and expected consequences."""
        features = dict(world_state.data)
        features.update(expected_consequences.predictions)
        return LatentState(features=features)

    def _project_future_state(self, latent_state: LatentState) -> FutureState:
        """PLACEHOLDER LOGIC: carry latent features forward unchanged."""
        return FutureState(horizon=1.0, data=dict(latent_state.features))
