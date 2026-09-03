"""PredictionError: the discrepancy between expectation and actual outcome."""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class PredictionError:
    """The difference between an expected/future state and what was observed.

    Attributes:
        details: Arbitrary discrepancy values, keyed by name.
    """

    details: Dict[str, Any] = field(default_factory=dict)
