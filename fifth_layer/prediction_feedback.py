"""Evaluate due position forecasts against observations of the same track."""

from collections import deque
from math import hypot
from fifth_layer.perception.temporal import box_center


class PredictionFeedback:
    def __init__(self, tolerance=0.25, capacity=256):
        self.tolerance = tolerance
        self.capacity = capacity
        self.reset()

    def reset(self):
        self.pending = deque(maxlen=self.capacity)
        self.history = deque(maxlen=self.capacity)

    def record(self, timestamp, track_id, predicted_center, horizon=1.0):
        if track_id is None or predicted_center is None:
            return
        # At most one outstanding forecast per track avoids frame-rate bias.
        if any(item["track_id"] == track_id for item in self.pending):
            return
        self.pending.append(dict(track_id=track_id, due=timestamp + horizon,
                                 center=tuple(predicted_center)))

    def observe(self, timestamp, detections, width, height):
        visible = {item["track_id"]: item for item in detections if "track_id" in item}
        remaining = deque(maxlen=self.capacity)
        results = []
        for forecast in self.pending:
            if timestamp < forecast["due"]:
                remaining.append(forecast)
                continue
            detection = visible.get(forecast["track_id"])
            result = dict(track_id=forecast["track_id"], due=forecast["due"], observed_at=timestamp)
            if timestamp - forecast["due"] > self.tolerance:
                result["status"] = "expired"
            elif detection is None:
                # Absence is not proof that predicted occlusion occurred.
                remaining.append(forecast)
                continue
            else:
                center = box_center(detection)
                error = hypot(center[0] - forecast["center"][0], center[1] - forecast["center"][1])
                result.update(status="evaluated", error_pixels=error,
                              normalized_error=error / max(hypot(width, height), 1.0))
            results.append(result)
            self.history.append(result)
        self.pending = remaining
        return results
