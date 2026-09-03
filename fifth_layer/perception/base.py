"""Base interface for Fifth Layer perception modules."""

from abc import ABC, abstractmethod

from fifth_layer.world_state import WorldState


class BasePerception(ABC):
    """Abstract interface for converting sensory input into WorldState."""

    @abstractmethod
    def perceive(self, source) -> WorldState:
        """Convert an input source into a WorldState."""
        raise NotImplementedError