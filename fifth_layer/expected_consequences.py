"""ExpectedConsequences: predicted effects/outcomes expected from a state."""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ExpectedConsequences:
    """Predicted consequences expected to follow from a state or action.

    Attributes:
        predictions: Arbitrary predicted outcomes, keyed by name.
    """

    predictions: Dict[str, Any] = field(default_factory=dict)
