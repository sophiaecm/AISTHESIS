"""Bounded position estimates, never detector evidence or confidence."""
from copy import deepcopy
from math import hypot
from fifth_layer.perception.temporal import _box_to_xywh

NORMAL_PREDICTION_TIMEOUT_SECONDS = 1.5
OCCLUSION_PREDICTION_TIMEOUT_SECONDS = 2.5
MAX_RELIABLE_VELOCITY_INTERVAL_SECONDS = 1.5
MAX_NORMALIZED_PREDICTION_SPEED = 0.5
MAX_RELIABLE_SIZE_CHANGE = 0.25
MIN_VELOCITY_INTERVAL_SECONDS = 0.02


def reliable_velocity(old, bbox, dt, velocity, width, height):
    if old is None or old['missed_frames']:
        return False
    old_w, old_h = old['bbox'][2]-old['bbox'][0], old['bbox'][3]-old['bbox'][1]
    w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
    return (MIN_VELOCITY_INTERVAL_SECONDS <= dt <= MAX_RELIABLE_VELOCITY_INTERVAL_SECONDS
            and hypot(*velocity) <= MAX_NORMALIZED_PREDICTION_SPEED * hypot(width, height)
            and abs(w-old_w) <= MAX_RELIABLE_SIZE_CHANGE * max(old_w, 1)
            and abs(h-old_h) <= MAX_RELIABLE_SIZE_CHANGE * max(old_h, 1))


def _overlaps(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def predict_missing(track, timestamp, width, height, visible=()):
    age = max(0.0, timestamp-track['last_seen_timestamp'])
    reliable = track.get('velocity_reliable', False)
    vx, vy = (track['velocity_x'], track['velocity_y']) if reliable else (0.0, 0.0)
    cx, cy = track['center'][0]+vx*age, track['center'][1]+vy*age
    box = track['bbox']
    w, h = min(width, box[2]-box[0]), min(height, box[3]-box[1])
    x = min(max(0.0, cx-w/2), width-w)
    y = min(max(0.0, cy-h/2), height-h)
    bbox = [x, y, x+w, y+h]
    sweep = [min(box[0], bbox[0]), min(box[1], bbox[1]),
             max(box[2], bbox[2]), max(box[3], bbox[3])]
    possible = track.get('possible_occlusion', False)
    for detection in visible:
        if detection.get('track_id') == track['track_id']:
            continue
        left, top, bw, bh = _box_to_xywh(detection)
        if _overlaps(sweep, [left, top, left+bw, top+bh]):
            possible = True
    limit = OCCLUSION_PREDICTION_TIMEOUT_SECONDS if possible else NORMAL_PREDICTION_TIMEOUT_SECONDS
    if age > limit:
        return None
    return dict(track_id=track['track_id'], class_name=track['class_name'],
                observation_state='predicted', is_predicted=True,
                prediction_age_seconds=age, predicted_center=[cx, cy], predicted_bbox=bbox,
                position_uncertainty=2.0 + age * (10.0 if reliable else 25.0),
                position_uncertainty_units='pixels', velocity_reliable=reliable,
                possible_occlusion=possible, evidence='possible_occlusion' if possible else None,
                prediction_valid_until=track['last_seen_timestamp']+limit)


def prediction_context(data, timestamp=None):
    """Optional reasoner context; never merged into observed hypotheses/counts."""
    items = data.get('predicted_tracks', [])
    if not isinstance(items, list):
        return []
    return [deepcopy(p) for p in items if isinstance(p, dict)
            and p.get('is_predicted') is True and p.get('observation_state') == 'predicted'
            and (timestamp is None or timestamp <= p.get('prediction_valid_until', -1))]
