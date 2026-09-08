"""Bounded, constant-velocity association of visible detections."""

from copy import deepcopy
from math import hypot
from fifth_layer.perception.temporal import box_center


class ObjectTracker:
    def __init__(self, max_age=1.5, max_distance=0.12):
        self.max_age = max_age
        self.max_distance = max_distance
        self.reset()

    def reset(self):
        self.tracks = {}
        self.next_id = 0
        self.last_timestamp = None

    def update(self, detections, timestamp, width, height):
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            self.reset()
        self.last_timestamp = timestamp
        self.tracks = {key: value for key, value in self.tracks.items()
                       if timestamp - value["timestamp"] <= self.max_age}
        result = deepcopy(detections)
        diagonal = hypot(width, height)
        pairs = []
        for index, detection in enumerate(result):
            center = box_center(detection)
            for track_id, track in self.tracks.items():
                if detection.get("class_name") != track["class_name"]:
                    continue
                dt = timestamp - track["timestamp"]
                predicted = [track["center"][i] + track["velocity"][i] * dt for i in (0, 1)]
                distance = hypot(center[0] - predicted[0], center[1] - predicted[1])
                if diagonal > 0 and distance / diagonal <= self.max_distance:
                    pairs.append((distance, index, track_id))
        assigned = {}
        used = set()
        for _, index, track_id in sorted(pairs):
            if index not in assigned and track_id not in used:
                assigned[index] = track_id
                used.add(track_id)
        for index, detection in enumerate(result):
            track_id = assigned.get(index)
            if track_id is None:
                track_id = self.next_id
                self.next_id += 1
            center = box_center(detection)
            old = self.tracks.get(track_id)
            velocity = (0.0, 0.0)
            if old is not None:
                dt = timestamp - old["timestamp"]
                velocity = tuple((center[i] - old["center"][i]) / dt for i in (0, 1))
            self.tracks[track_id] = dict(center=center, velocity=velocity,
                                        timestamp=timestamp, class_name=detection.get("class_name"))
            detection["track_id"] = track_id
        # Missing tracks remain internal; never synthesize visible detections.
        return result
