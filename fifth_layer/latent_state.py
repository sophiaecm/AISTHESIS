"""LatentState: an internal/hidden representation derived from a WorldState."""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class LatentState:
    """An internal, hidden representation derived from observed state.

    Attributes:
        features: Arbitrary latent features, keyed by name.
    """

    features: Dict[str, Any] = field(default_factory=dict)
