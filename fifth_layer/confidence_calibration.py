"""Bounded session calibration of prediction outcomes, never detector evidence."""
from collections import OrderedDict
from copy import deepcopy
from math import isfinite
from threading import RLock

CORRECT_WEIGHT = 1.0
PARTIAL_WEIGHT = 0.5
INCORRECT_WEIGHT = 0.0
ALPHA = 2.0
BETA = 2.0
MIN_CALIBRATION_SAMPLES = 5
MAX_RELATIVE_EFFECT = 0.20
MAX_BUCKETS = 256
MAX_PROCESSED_IDS = 4096
CALIBRATION_TTL_SECONDS = 300.0
OUTCOME_WEIGHTS = {'correct': CORRECT_WEIGHT, 'partially_correct': PARTIAL_WEIGHT,
                   'incorrect': INCORRECT_WEIGHT}


class ConfidenceCalibrationMemory:
    def __init__(self, max_buckets=MAX_BUCKETS, max_ids=MAX_PROCESSED_IDS,
                 ttl=CALIBRATION_TTL_SECONDS, event_logger=None):
        if max_buckets < 3 or max_ids < 1 or ttl <= 0:
            raise ValueError('Invalid calibration memory limits')
        self.max_buckets, self.max_ids, self.ttl = max_buckets, max_ids, ttl
        self.event_logger = event_logger
        self.lock = RLock()
        self.reset()

    def reset(self):
        with self.lock:
            self.buckets, self.processed = OrderedDict(), OrderedDict()
            self.now = self.closed_through = float('-inf')

    def _cleanup(self, timestamp):
        self.now = max(self.now, timestamp)
        self.closed_through = max(self.closed_through, self.now-self.ttl)
        self.buckets = OrderedDict((k, b) for k, b in self.buckets.items()
            if b['last_updated_timestamp'] > self.now-self.ttl)
        self.processed = OrderedDict((k, t) for k, t in self.processed.items()
            if t > self.closed_through)

    @staticmethod
    def _keys(source, prediction_type, class_name):
        return [(source, prediction_type, class_name), (source, prediction_type), (source,)]

    def update(self, result):
        """Only finalized evaluable outcomes. Watermark prevents replay after eviction."""
        with self.lock:
            if result['status'] not in OUTCOME_WEIGHTS:
                return False
            timestamp = result['evaluated_timestamp']
            self._cleanup(timestamp)
            pid = result['prediction_id']
            if pid in self.processed or timestamp <= self.closed_through:
                return False
            if len(self.processed) >= self.max_ids:
                self.closed_through = max(self.closed_through, min(self.processed.values()))
                self._cleanup(self.now)
                if timestamp <= self.closed_through:
                    return False
            self.processed[pid] = timestamp
            p = result['prediction']
            weight = OUTCOME_WEIGHTS[result['status']]
            for key in self._keys(p['source'], p.get('prediction_type', 'position'), p.get('class_name')):
                b = self.buckets.pop(key, dict(total_evaluable=0, correct_count=0,
                    partial_count=0, incorrect_count=0, weighted_success=0.0,
                    reliability=ALPHA/(ALPHA+BETA), last_updated_timestamp=timestamp))
                b['total_evaluable'] += 1
                count = {'correct': 'correct_count', 'partially_correct': 'partial_count',
                         'incorrect': 'incorrect_count'}[result['status']]
                b[count] += 1
                b['weighted_success'] += weight
                b['reliability'] = (ALPHA+b['weighted_success'])/(ALPHA+BETA+b['total_evaluable'])
                b['last_updated_timestamp'] = max(b['last_updated_timestamp'], timestamp)
                self.buckets[key] = b
            while len(self.buckets) > self.max_buckets:
                self.buckets.popitem(last=False)
            if self.event_logger:
                self.event_logger('CONFIDENCE_CALIBRATION', prediction_id=pid,
                    source=p['source'], prediction_type=p.get('prediction_type', 'position'),
                    class_name=p.get('class_name'), status=result['status'],
                    issued_confidence={k: p.get(k) for k in ('raw_confidence',
                        'calibration_reliability', 'calibrated_confidence', 'calibration_samples')},
                    statistics=self.statistics())
            return True

    def calibrate(self, raw, source, prediction_type, class_name, timestamp):
        with self.lock:
            if not isfinite(raw) or not 0 <= raw <= 1:
                raise ValueError('Raw confidence must be finite and in [0, 1]')
            self._cleanup(timestamp)
            keys = self._keys(source, prediction_type, class_name)
            key = next((k for k in keys if self.buckets.get(k, {}).get('total_evaluable', 0)
                        >= MIN_CALIBRATION_SAMPLES), None)
            # No cross-source empirical fallback: the global prior is fixed.
            n = self.buckets.get((source,), {}).get('total_evaluable', 0)
            reliability = ALPHA/(ALPHA+BETA)
            if key is not None:
                n = self.buckets[key]['total_evaluable']
                reliability = self.buckets[key]['reliability']
            multiplier = 1 + MAX_RELATIVE_EFFECT*(2*reliability-1)
            return dict(raw_confidence=raw, calibration_reliability=reliability,
                calibrated_confidence=min(1.0, max(0.0, raw*multiplier)),
                calibration_samples=n, calibration_bucket=list(key) if key else ['global_prior'])

    def statistics(self):
        with self.lock:
            return [dict(bucket=list(k), **deepcopy(v)) for k, v in self.buckets.items()]

    def advance(self, timestamp):
        with self.lock:
            self._cleanup(timestamp)


def calibration_overlay(result):
    """Show issuance-time values; never recalibrate the evaluated forecast."""
    if not result or 'raw_confidence' not in result.get('prediction', {}):
        return ''
    p = result['prediction']
    n = p['calibration_samples']
    if n < MIN_CALIBRATION_SAMPLES:
        return f'CAL: collecting evidence | n={n}/{MIN_CALIBRATION_SAMPLES}'
    return (f"CAL: raw {p['raw_confidence']:.2f} -> calibrated "
            f"{p['calibrated_confidence']:.2f} | n={n}")
