"""WorldState: the observed state of the world at a point in time."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class WorldState:
    """A snapshot of the observed/raw state of the world.

    Attributes:
        timestamp: When this state was observed, if known.
        data: Arbitrary observed values (e.g. sensor readings, features).
            Optional predicted_tracks contains bounded position estimates only;
            it is separate from detections/accepted_detections and object counts.
            Optional evaluation_predictions and prediction_evaluations hold
            measurement-only forecast records and finalized comparison results.
            Forecast records optionally preserve raw_confidence,
            calibration_reliability and calibrated_confidence at issuance.
            These fields are prediction metadata, never detector confidence.
            Optional fast_scene_narration contains a structured local description;
            fast_scene_description/source/timestamp/latency_ms/detail_level/mode
            are display metadata independent of deep scene state.
    """

    timestamp: Optional[float] = None
    data: Dict[str, Any] = field(default_factory=dict)
