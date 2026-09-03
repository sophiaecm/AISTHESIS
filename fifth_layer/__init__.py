"""Fifth Layer Engine.

A minimal, modular scaffold for AISTHESIS's predictive inference core.

This package currently defines five core data concepts:

- :class:`WorldState`
- :class:`ExpectedConsequences`
- :class:`LatentState`
- :class:`FutureState`
- :class:`PredictionError`

and a pipeline that connects them:

- :class:`FifthLayerEngine`

As of v0.3, FifthLayerEngine is model-independent: all inference is
delegated to a pluggable reasoner implementing :class:`BaseReasoner`.
:class:`PlaceholderReasoner` is the default, used only to validate the
architecture. No LLM adapters are implemented yet.
"""

from fifth_layer.reasoners.probabilistic import ProbabilisticReasoner
from fifth_layer.world_state import WorldState
from fifth_layer.expected_consequences import ExpectedConsequences
from fifth_layer.latent_state import LatentState
from fifth_layer.future_state import FutureState
from fifth_layer.prediction_error import PredictionError
from fifth_layer.engine import FifthLayerEngine
from fifth_layer.reasoners import (
    BaseReasoner,
    PlaceholderReasoner,
    RuleBasedReasoner,
    PhysicsReasoner,
    ProbabilisticReasoner,
)

__all__ = [
    "WorldState",
    "ExpectedConsequences",
    "LatentState",
    "FutureState",
    "PredictionError",
    "FifthLayerEngine",
    "BaseReasoner",
    "PlaceholderReasoner",
    "RuleBasedReasoner",
    "PhysicsReasoner",
    "ProbabilisticReasoner",
]

