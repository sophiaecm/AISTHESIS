"""BaseReasoner: the model-independent inference interface.

Any reasoning strategy (rule-based, physics-based, learned, sensor-fusion,
probabilistic, or an optional future LLM adapter) implements this
interface. The Fifth Layer core never depends on a specific strategy -
it only depends on this abstract contract.
"""

from abc import ABC, abstractmethod

from fifth_layer.world_state import WorldState
from fifth_layer.expected_consequences import ExpectedConsequences
from fifth_layer.latent_state import LatentState
from fifth_layer.future_state import FutureState


class BaseReasoner(ABC):
    """Abstract interface for pluggable reasoning strategies.

    Implementations decide how to turn a WorldState into
    ExpectedConsequences, a LatentState, and finally a FutureState.
    """

    @abstractmethod
    def infer_expected_consequences(self, world_state: WorldState) -> ExpectedConsequences:
        """Infer the ExpectedConsequences of a given WorldState."""
        raise NotImplementedError

    @abstractmethod
    def infer_latent_state(
        self,
        world_state: WorldState,
        expected_consequences: ExpectedConsequences,
    ) -> LatentState:
        """Infer a LatentState from a WorldState and ExpectedConsequences."""
        raise NotImplementedError

    @abstractmethod
    def infer_future_state(self, latent_state: LatentState) -> FutureState:
        """Infer a FutureState from a LatentState."""
        raise NotImplementedError
