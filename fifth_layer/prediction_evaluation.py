"""Session-only measurement of forecasts; never feeds confidence or reasoning."""
from collections import deque
from copy import deepcopy
from math import hypot
from threading import RLock
from uuid import uuid4

from fifth_layer.perception.tracking import _bbox, _iou

CORRECT_NORMALIZED_ERROR = 0.05
PARTIAL_NORMALIZED_ERROR = 0.12
MEANINGFUL_BBOX_IOU = 0.30
STATIONARY_NORMALIZED_DISPLACEMENT = 0.01
DIRECTION_MIN_COSINE = 0.7071067811865476  # 45 degrees
EVALUATION_GRACE_SECONDS = 0.25
MEMORY_TTL_SECONDS = 60.0
MEMORY_MAX_RECORDS = 512
SOURCES = {'temporal_live', 'temporal_deep', 'occlusion_track'}


def compare_prediction(prediction, observation, evaluated_timestamp):
    """Compare explicit observed evidence in the forecast's image geometry."""
    p, o = prediction, observation
    result = {key: p[key] for key in (
        'prediction_id', 'track_id', 'prediction_created_timestamp',
        'prediction_horizon_seconds', 'target_timestamp', 'source')}
    result.update(status='unevaluable', center_error_pixels=None,
                  normalized_center_error=None, bbox_iou=None, direction_match=None,
                  timing_error_seconds=None, evaluated_timestamp=evaluated_timestamp,
                  evaluation_reason='no_suitable_observation', prediction=deepcopy(p),
                  observation=deepcopy(o))
    if evaluated_timestamp < p['target_timestamp']:
        raise ValueError('Forecast is not due')
    if (o is None or o.get('observation_state') != 'observed' or o.get('is_predicted')
            or o.get('track_id') != p['track_id']
            or o['observed_timestamp'] <= p['prediction_created_timestamp']
            or o['observed_timestamp'] > evaluated_timestamp
            or abs(o['observed_timestamp'] - p['target_timestamp']) > EVALUATION_GRACE_SECONDS):
        return result
    if o.get('image_size', p['image_size']) != p['image_size']:
        result['evaluation_reason'] = 'image_geometry_changed'
        return result
    diagonal = hypot(*p['image_size'])
    error = hypot(*(a-b for a, b in zip(p['predicted_center'], o['observed_center'])))
    normalized = error / diagonal
    origin = p['origin_center']
    expected = [p['predicted_center'][i]-origin[i] for i in (0, 1)]
    actual = [o['observed_center'][i]-origin[i] for i in (0, 1)]
    small_p = hypot(*expected) / diagonal <= STATIONARY_NORMALIZED_DISPLACEMENT
    small_o = hypot(*actual) / diagonal <= STATIONARY_NORMALIZED_DISPLACEMENT
    # Deadband: stationary agrees with stationary; avoid noisy angle labels.
    direction = (small_p and small_o) if small_p or small_o else (
        sum(a*b for a, b in zip(expected, actual)) /
        (hypot(*expected)*hypot(*actual)) >= DIRECTION_MIN_COSINE)
    iou = _iou(p['predicted_bbox'], o['observed_bbox'])
    status = ('correct' if normalized <= CORRECT_NORMALIZED_ERROR and direction
              else 'partially_correct' if normalized <= PARTIAL_NORMALIZED_ERROR or iou >= MEANINGFUL_BBOX_IOU
              else 'incorrect')
    result.update(status=status, center_error_pixels=error, normalized_center_error=normalized,
                  bbox_iou=iou, direction_match=direction,
                  timing_error_seconds=o['observed_timestamp']-p['target_timestamp'],
                  evaluation_reason='same_track_observed_geometry')
    return result


def evaluation_overlay(result):
    if not result:
        return ''
    error = result['center_error_pixels']
    return 'EVAL: ' + result['status'] + (f' | error {error:.1f}px' if error is not None else '')


class PredictionEvaluationMemory:
    """Bound every collection; finalize after the symmetric time window closes.

    Admission is internal and IDs are generated here, so evicted results cannot
    be resubmitted. Late deep results use retained observation-time history.
    All public operations are serialized across camera and background threads.
    """
    def __init__(self, capacity=MEMORY_MAX_RECORDS, ttl=MEMORY_TTL_SECONDS, event_logger=None):
        if capacity < 1 or ttl <= EVALUATION_GRACE_SECONDS:
            raise ValueError('Invalid memory limits')
        self.capacity, self.ttl, self.event_logger = capacity, ttl, event_logger
        self.lock = RLock()
        self.reset()

    def reset(self):
        with self.lock:
            self.pending = deque()
            self.history = deque(maxlen=self.capacity)
            self.observations = deque(maxlen=self.capacity)
            self.now = float('-inf')

    def _finish(self, p, observation=None, expired=False):
        result = compare_prediction(p, observation, max(self.now, p['target_timestamp']))
        if expired:
            result.update(status='expired', evaluation_reason='memory_limit_or_ttl')
        self.history.append(result)
        if self.event_logger:
            self.event_logger('PREDICTION_EVALUATION', **result)
        return result

    def advance(self, timestamp):
        with self.lock:
            self.now = max(self.now, timestamp)
            self.history = deque((r for r in self.history if self.now-r['evaluated_timestamp'] <= self.ttl), maxlen=self.capacity)
            self.observations = deque((o for o in self.observations if self.now-o['observed_timestamp'] <= self.ttl), maxlen=self.capacity)
            remaining, results = deque(), []
            for p in self.pending:
                if self.now < p['target_timestamp'] + EVALUATION_GRACE_SECONDS:
                    remaining.append(p)
                    continue
                candidates = [o for o in self.observations
                              if o['track_id'] == p['track_id']
                              and o['observed_timestamp'] > p['prediction_created_timestamp']
                              and o['image_size'] == p['image_size']
                              and abs(o['observed_timestamp']-p['target_timestamp']) <= EVALUATION_GRACE_SECONDS]
                closest = min(candidates, key=lambda o: abs(o['observed_timestamp']-p['target_timestamp']), default=None)
                results.append(self._finish(p, closest, self.now-p['target_timestamp'] > self.ttl))
            self.pending = remaining
            return results

    def observe(self, timestamp, detections, width, height):
        with self.lock:
            for d in detections:
                if d.get('observation_state') != 'observed' or d.get('is_predicted') or d.get('track_id') is None:
                    continue
                bbox = _bbox(d)
                self.observations.append(dict(track_id=d['track_id'], observed_bbox=bbox,
                    observed_center=[(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2],
                    observed_timestamp=timestamp, observation_state='observed', image_size=[width, height]))
            return self.advance(timestamp)

    def record_state(self, state, predictions, source):
        """Register actual reasoner trajectories, never rematch by class."""
        with self.lock:
            if source not in SOURCES:
                raise ValueError('Unknown prediction source')
            data = state.data
            width, height = data.get('image_width', 0), data.get('image_height', 0)
            if state.timestamp is None or width <= 0 or height <= 0:
                return []
            records = []
            if source == 'occlusion_track':
                for p in data.get('predicted_tracks', []):
                    records.append((p, [dict(center=p['predicted_center'], bbox=p['predicted_bbox'], horizon_seconds=0)]))
            else:
                track_id = predictions.get('track_id')
                detection = next((d for d in data.get('detections', [])
                                  if track_id is not None and d.get('track_id') == track_id
                                  and d.get('observation_state') == 'observed'), None)
                if detection:
                    records.append((detection, predictions.get('trajectory', [])))
            added = []
            for d, trajectory in records:
                for point in trajectory:
                    horizon = point['horizon_seconds']
                    if not 0 <= horizon <= self.ttl - EVALUATION_GRACE_SECONDS:
                        continue
                    if any(p['track_id'] == d['track_id'] and p['source'] == source
                           and p['prediction_horizon_seconds'] == horizon for p in self.pending):
                        continue
                    origin_box = d.get('origin_bbox') or _bbox(d)
                    origin = d.get('origin_center') or [(origin_box[0]+origin_box[2])/2, (origin_box[1]+origin_box[3])/2]
                    center = point['center']
                    dx, dy = center[0]-origin[0], center[1]-origin[1]
                    created = d.get('prediction_created_timestamp', state.timestamp) if source == 'occlusion_track' else state.timestamp
                    target = state.timestamp+horizon
                    p = dict(prediction_id=uuid4().hex, track_id=d['track_id'], class_name=d.get('class_name'),
                             predicted_center=list(center), predicted_bbox=point.get('bbox') or [origin_box[0]+dx, origin_box[1]+dy, origin_box[2]+dx, origin_box[3]+dy],
                             predicted_motion_direction=predictions.get('motion_state', d.get('predicted_motion_direction', 'stationary')),
                             prediction_created_timestamp=created, prediction_horizon_seconds=target-created,
                             target_timestamp=target, position_uncertainty_at_prediction=d.get('position_uncertainty', 0.0),
                             source=source, origin_center=list(origin), image_size=[width, height])
                    if len(self.pending) >= self.capacity:
                        # Discard admission rather than prematurely evaluating a future forecast.
                        continue
                    self.pending.append(p)
                    added.append(deepcopy(p))
            return added

    def latest(self, timestamp):
        with self.lock:
            self.advance(timestamp)
            return deepcopy(self.history[-1]) if self.history else None

    def results_for_source(self, source):
        with self.lock:
            return deepcopy([r for r in self.history if r['source'] == source])
