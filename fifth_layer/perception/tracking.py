"""Local deterministic identity association; missing boxes stay last-observed."""

from fifth_layer.perception.track_prediction import predict_missing, reliable_velocity
from copy import deepcopy
from math import hypot, isfinite
from fifth_layer.perception.temporal import box_center, _box_to_xywh


DEFAULT_MAX_AGE_SECONDS = 1.5
DEFAULT_MAX_CENTER_DISTANCE = 0.12
DEFAULT_MAX_MISSED_FRAMES = 8
MIN_ASSOCIATION_IOU = 0.30
MIN_CONTAINMENT_OVERLAP = 0.80
MIN_AREA_RATIO = 0.40
PREDICTED_DISTANCE_WEIGHT = 0.65
CENTER_DISTANCE_WEIGHT = 0.20
IOU_WEIGHT = 0.10
ELAPSED_WEIGHT = 0.05


def _bbox(detection):
    x, y, w, h = _box_to_xywh(detection)
    return (x, y, x + w, y + h)


def _iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1]))
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - intersection)
    return intersection / union if union > 0 else 0.0


class TrackMemory:
    """Tracks age once per detector observation, including empty observations.

    Skipped inference frames must not call update. Expired records are removed
    from tracks and exposed in expired_tracks only until the next update.
    Velocity prediction is used only for association, never as an observation.
    IDs are unique within a session (until explicit reset).
    preserve_observation_continuity allows strong overlap to bridge slow
    inference intervals only when no missing observation has been recorded.
    observation_expiration requires BOTH missed observations and elapsed time,
    and removes time-only rejection before matching. min_confidence filters
    accepted observations before they can consume identities.
    The default retains the legacy timeout contract. debug records candidate
    decisions for the latest update only, without console output.
    """

    def __init__(self, max_age=DEFAULT_MAX_AGE_SECONDS,
                 max_distance=DEFAULT_MAX_CENTER_DISTANCE,
                 max_missed_frames=DEFAULT_MAX_MISSED_FRAMES,
                 preserve_observation_continuity=False, debug=False,
                 observation_expiration=False, min_confidence=None, event_logger=None,
                 predict_missing_tracks=False):
        if (not isfinite(max_age) or max_age <= 0
                or not isfinite(max_distance) or max_distance <= 0
                or not isinstance(max_missed_frames, int) or max_missed_frames < 0):
            raise ValueError("Invalid tracking limits")
        self.preserve_observation_continuity = preserve_observation_continuity
        self.predict_missing_tracks = predict_missing_tracks
        self.observation_expiration = observation_expiration
        self.min_confidence = min_confidence
        self.event_logger = event_logger
        self.debug = debug
        self.match_debug = []
        self.max_age = max_age
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames
        self.reset()

    def reset(self):
        self.tracks = {}
        self.expired_tracks = []
        self.next_id = 0
        self.predicted_tracks = []
        self.last_timestamp = None
        self.match_debug = []

    def _event(self, name, **fields):
        if self.event_logger is not None:
            self.event_logger(name, **fields)

    def _accepted(self, detection):
        if (detection.get("accepted") is False
                or any(detection.get(flag) for flag in ("rejected", "uncertain", "predicted", "is_predicted"))
                or detection.get("status") in {"rejected", "uncertain", "predicted"}):
            return False
        if self.min_confidence is not None:
            confidence = float(detection.get("confidence", 0.0))
            return isfinite(confidence) and confidence >= self.min_confidence
        return True

    def _expire(self, track_id, reason):
        track = self.tracks.pop(track_id)
        track["status"] = "expired"
        self._event("TRACK_EXPIRE", track_id=track_id, exact_expiration_reason=reason)
        self.expired_tracks.append(deepcopy(track))

    def update(self, detections, timestamp, width, height):
        if not isfinite(timestamp) or width <= 0 or height <= 0:
            raise ValueError("Finite timestamp and positive image dimensions required")
        if self.last_timestamp is not None and timestamp < self.last_timestamp:
            raise ValueError("Observation timestamps must not go backwards")
        result = deepcopy([d for d in detections if self._accepted(d)])
        geometry = [(box_center(d), _bbox(d)) for d in result]
        prior_predictions = {p["track_id"] for p in self.predicted_tracks}
        self.predicted_tracks = []
        self.last_timestamp = timestamp
        self.expired_tracks = []
        self.match_debug = []
        for track in self.tracks.values():
            track["age_frames"] += 1

        diagonal = hypot(width, height)
        pairs = []
        for index, detection in enumerate(result):
            center, bbox = geometry[index]
            for track_id, track in self.tracks.items():
                dt = timestamp - track["last_seen_timestamp"]
                predicted = (track["center"][0] + track["velocity_x"] * dt,
                             track["center"][1] + track["velocity_y"] * dt)
                distance = hypot(center[0] - predicted[0], center[1] - predicted[1]) / diagonal
                observed_distance = hypot(center[0] - track["center"][0],
                                          center[1] - track["center"][1]) / diagonal
                travel = hypot(track["velocity_x"], track["velocity_y"]) * dt / diagonal
                iou = _iou(bbox, track["bbox"])
                old_box = track["bbox"]
                area = max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])
                old_area = max(0, old_box[2] - old_box[0]) * max(0, old_box[3] - old_box[1])
                intersection = (max(0, min(bbox[2], old_box[2]) - max(bbox[0], old_box[0]))
                                * max(0, min(bbox[3], old_box[3]) - max(bbox[1], old_box[1])))
                overlap = intersection / min(area, old_area) if min(area, old_area) > 0 else 0
                area_ratio = min(area, old_area) / max(area, old_area) if max(area, old_area) > 0 else 0
                overlap_ok = (iou >= MIN_ASSOCIATION_IOU or
                              (overlap >= MIN_CONTAINMENT_OVERLAP and area_ratio >= MIN_AREA_RATIO))
                center_ok = (distance <= self.max_distance and
                             observed_distance <= self.max_distance + travel)
                overdue = dt > self.max_age
                # An inference gap is not evidence of disappearance. Only a
                # previously visible track with strong overlap may bridge it.
                continuity = (self.preserve_observation_continuity
                              and track["missed_frames"] == 0 and overlap_ok)
                if detection.get("class_name", "unknown") != track["class_name"]:
                    reason = "class_mismatch"
                elif overdue and not continuity and not self.observation_expiration:
                    reason = "missing_timeout" if track["missed_frames"] else "timeout"
                elif not (center_ok or overlap_ok):
                    reason = "geometry_gate"
                else:
                    reason = "candidate_center" if center_ok else "candidate_overlap"
                    if overdue:
                        reason = "candidate_observation_continuity"
                record = dict(old_track_id=track_id, detection_index=index, iou=iou,
                              center_distance=observed_distance * diagonal,
                              predicted_center_distance=distance * diagonal,
                              elapsed_time=dt, reason=reason, accepted=False)
                if self.debug or self.event_logger is not None:
                    self.match_debug.append(record)
                if not reason.startswith("candidate_"):
                    self._event("TRACK_MATCH", track_id=track_id,
                                class_name=track["class_name"], IoU=iou,
                                center_distance=observed_distance * diagonal,
                                predicted_center_distance=distance * diagonal,
                                elapsed_time=dt, accepted_reason=reason)
                    continue
                score = (PREDICTED_DISTANCE_WEIGHT * min(distance / self.max_distance, 1.0)
                         + CENTER_DISTANCE_WEIGHT * min(observed_distance / (self.max_distance + travel), 1.0)
                         + IOU_WEIGHT * (1 - iou)
                         + ELAPSED_WEIGHT * min(dt / self.max_age, 1.0))
                pairs.append((score, index, track_id, record))
        assigned = {}
        used = set()
        for _, index, track_id, record in sorted(pairs, key=lambda pair: pair[:3]):
            if index not in assigned and track_id not in used:
                assigned[index] = track_id
                used.add(track_id)
                record["accepted"] = True
                record["reason"] = record["reason"].replace("candidate_", "accepted_")
            else:
                record["reason"] = "one_to_one_conflict"
        for _, index, track_id, record in pairs:
            self._event("TRACK_MATCH", track_id=track_id,
                        class_name=self.tracks[track_id]["class_name"], IoU=record["iou"],
                        center_distance=record["center_distance"],
                        predicted_center_distance=record["predicted_center_distance"],
                        elapsed_time=record["elapsed_time"], accepted_reason=record["reason"])
        for track_id, track in list(self.tracks.items()):
            if track_id not in used:
                track["missed_frames"] += 1
                track["status"] = "temporarily_missing"
                prediction = (predict_missing(track, timestamp, width, height, result)
                              if self.predict_missing_tracks else None)
                if prediction is not None:
                    track["possible_occlusion"] = prediction["possible_occlusion"]
                    self.predicted_tracks.append(prediction)
                elapsed = timestamp - track["last_seen_timestamp"]
                self._event("TRACK_MISS", track_id=track_id,
                            missed_frames=track["missed_frames"], elapsed_time=elapsed)
                missed_limit = track["missed_frames"] > self.max_missed_frames
                time_limit = elapsed > self.max_age
                should_expire = ((missed_limit and time_limit) if self.observation_expiration
                                 else (missed_limit or time_limit))
                if should_expire and not (self.predict_missing_tracks and prediction is not None):
                    reason = ("missed_observations_and_min_elapsed" if self.observation_expiration
                              else "missed_limit" if missed_limit else "wall_clock_timeout")
                    self._expire(track_id, reason)
        for index, detection in enumerate(result):
            track_id = assigned.get(index)
            if track_id is None:
                track_id = self.next_id
                self.next_id += 1
                candidates = [r for r in self.match_debug if r["detection_index"] == index]
                reason = (",".join(sorted({r["reason"] for r in candidates}))
                          if candidates else "no_existing_track")
                self._event("TRACK_CREATE", new_track_id=track_id,
                            class_name=detection.get("class_name", "unknown"),
                            confidence=detection.get("confidence", 0.0),
                            exact_creation_reason=reason)
            center, bbox = geometry[index]
            old = self.tracks.get(track_id)
            if old is not None and track_id in prior_predictions:
                estimate = predict_missing(old, timestamp, width, height)
                if estimate is not None:
                    detection["reappearance_error_pixels"] = hypot(
                        center[0]-estimate["predicted_center"][0],
                        center[1]-estimate["predicted_center"][1])
            velocity = (0.0, 0.0)
            if old is not None:
                dt = timestamp - old["last_seen_timestamp"]
                velocity = (old["velocity_x"], old["velocity_y"])
                if dt > 0:
                    velocity = tuple((center[i] - old["center"][i]) / dt for i in (0, 1))
            self.tracks[track_id] = dict(
                track_id=track_id, class_name=detection.get("class_name", "unknown"),
                bbox=bbox, center=center, confidence=float(detection.get("confidence", 0.0)),
                velocity_x=velocity[0], velocity_y=velocity[1],
                first_seen_timestamp=old["first_seen_timestamp"] if old else timestamp,
                last_seen_timestamp=timestamp, age_frames=old["age_frames"] if old else 1,
                missed_frames=0, status="active",
                velocity_reliable=reliable_velocity(old, bbox, timestamp-old["last_seen_timestamp"] if old else 0,
                                                    velocity, width, height),
                # Compatibility with the original ObjectTracker record.
                velocity=velocity, timestamp=timestamp)
            detection["track_id"] = track_id
            if self.predict_missing_tracks:
                detection.update(observation_state="observed", is_predicted=False,
                                 prediction_age_seconds=0.0, predicted_center=None,
                                 predicted_bbox=None, position_uncertainty=0.0)
        return result


class ObjectTracker(TrackMemory):
    """Backward-compatible name for TrackMemory."""
