"""Reasoners: pluggable, model-independent inference strategies.

The Fifth Layer core depends only on :class:`BaseReasoner`. Concrete
strategies (e.g. :class:`PlaceholderReasoner`, and future strategies such
as rule-based, physics-based, learned, sensor-fusion, or probabilistic
reasoners, or optional LLM adapters) implement that interface and can be
swapped in without changing FifthLayerEngine.
"""

from fifth_layer.reasoners.base import BaseReasoner
from fifth_layer.reasoners.placeholder import PlaceholderReasoner
from fifth_layer.reasoners.rule_based import RuleBasedReasoner
from fifth_layer.reasoners.physics import PhysicsReasoner
__all__ = [
    "BaseReasoner",
    "PlaceholderReasoner",
    "RuleBasedReasoner",
    "PhysicsReasoner",
]
