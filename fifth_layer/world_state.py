"""WorldState: the observed state of the world at a point in time."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class WorldState:
    """A snapshot of the observed/raw state of the world.

    Attributes:
        timestamp: When this state was observed, if known.
        data: Arbitrary observed values (e.g. sensor readings, features).
    """

    timestamp: Optional[float] = None
    data: Dict[str, Any] = field(default_factory=dict)
