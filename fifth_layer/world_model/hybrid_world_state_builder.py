"""Deterministic, read-only assembly; no model loading or latent interpretation."""
from .hybrid_world_state import HybridWorldState


class HybridWorldStateBuilder:
    def build(self, physical_state, physics_constraints=None, learned_signal=None):
        return HybridWorldState(physical_state, physics_constraints, learned_signal)
