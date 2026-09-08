"""Immutable handoff of one observation to a background analysis worker."""

from dataclasses import dataclass
import json

from fifth_layer.world_state import WorldState


@dataclass(frozen=True)
class AnalysisSnapshot:
    timestamp: float
    pixels: bytes
    shape: tuple
    dtype: str
    state_json: str
    motion_json: str

    @classmethod
    def capture(cls, frame, world_state, stable_motion):
        # Serialized payloads share no mutable nested values with the producer.
        return cls(world_state.timestamp, frame.tobytes(), tuple(frame.shape),
                   frame.dtype.str, json.dumps(world_state.data),
                   json.dumps(stable_motion))

    def frame(self):
        import numpy as np
        return np.frombuffer(self.pixels, dtype=self.dtype).reshape(self.shape)

    def world_state(self):
        return WorldState(timestamp=self.timestamp, data=json.loads(self.state_json))

    def motion_evidence(self):
        return json.loads(self.motion_json)
