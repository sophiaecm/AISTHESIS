"""Fifth Layer Engine.

A minimal, modular scaffold for AISTHESIS's predictive inference core.

This package currently defines five core data concepts:

- :class:`WorldState`
- :class:`ExpectedConsequences`
- :class:`LatentState`
- :class:`FutureState`
- :class:`PredictionError`

and a v0.2 pipeline that connects them:

- :class:`FifthLayerEngine`

No LLM adapters or real inference/learning are implemented yet; the
engine's transformation logic is placeholder logic only.
"""

from fifth_layer.world_state import WorldState
from fifth_layer.expected_consequences import ExpectedConsequences
from fifth_layer.latent_state import LatentState
from fifth_layer.future_state import FutureState
from fifth_layer.prediction_error import PredictionError
from fifth_layer.engine import FifthLayerEngine

__all__ = [
    "WorldState",
    "ExpectedConsequences",
    "LatentState",
    "FutureState",
    "PredictionError",
    "FifthLayerEngine",
]
