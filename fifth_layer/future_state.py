"""FutureState: a projected future state."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class FutureState:
    """A predicted/projected future state.

    Attributes:
        horizon: How far ahead this projection looks, if known.
        data: Arbitrary projected values, keyed by name.
    """

    horizon: Optional[float] = None
    data: Dict[str, Any] = field(default_factory=dict)
