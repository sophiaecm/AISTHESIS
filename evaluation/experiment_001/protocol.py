"""Pure Experiment 001 contract checks. Only synthetic fixtures are run in Step 28A."""
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path


VERSION = 'experiment-001-protocol-v0.1'
CONDITIONS = ('A_VISUAL', 'B_VISUAL_PHYSICS', 'C_FULL', 'C_NO_ACOUSTIC', 'C_SHUFFLED_ACOUSTIC')


def _text(value):
    if type(value) is not str or not value.strip():
        raise ValueError('nonempty identifier required')


def _time(value):
    if type(value) not in (int, float) or not isfinite(value) or value < 0:
        raise ValueError('finite nonnegative source seconds required')


def _index(value, size):
    if type(value) is not int or not 0 <= value < size:
        raise ValueError('frame index outside sequence')


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


class _Record:
    def to_dict(self):
        return json.loads(self.to_json())

    def to_json(self):
        return _canonical(asdict(self))


@dataclass(frozen=True)
class ProtocolDefinition(_Record):
    manifest_json: str
    protocol_id: str = field(init=False)

    def __post_init__(self):
        data = json.loads(self.manifest_json)
        fixed = {
            'protocol_version': VERSION,
            'conditions': list(CONDITIONS),
            'target_semantics': 'appearance_actor_category_region_exact_v1',
            'event_onset_rule': 'first_visible_frame_in_monitoring_interval_confirmed_next_frame_v1',
            'onset_annotation': {'minimum_connected_visible_pixels_in_region': 16,
                'consecutive_confirming_frames': 2, 'masks': 'manual_source_frame_annotation_only',
                'onset_timestamp': 'first_frame_not_confirmation_frame'},
            'anticipation_rule': 'one_exact_category_region_commitment_per_frame_before_onset_within_horizon_v1',
            'horizon_seconds': 2.0,
            'units': 'seconds',
            'tta_definition': 'event_onset_timestamp minus earliest_valid_issuance_timestamp',
            'tta_missing': None,
            'negative_trial_policy': 'complete_annotation_no_target_event_is_negative_missing_is_not_negative',
            'false_anticipation_policy': 'resolved_incorrect_pre_event_commitments_and_negative_trial_commitments',
            'exclusion_policy': 'retain_missing_ambiguous_censored_timing_and_log_failures_separately_never_as_incorrect',
            'probability_metric_eligibility': {'brier_score': 'ineligible', 'nll': 'ineligible', 'ece': 'ineligible'},
            'shuffle_policy': 'split_cutoff_group_sorted_sequence_ids_next_rotation_no_self_singleton_unavailable_v1',
        }
        if not isinstance(data, dict) or any(k not in data or data[k] != v for k, v in fixed.items()):
            raise ValueError('unsupported protocol semantics')
        _time(data['horizon_seconds'])
        if not data.get('metric_identifiers') or any('overall' in x or 'composite' in x for x in data['metric_identifiers']):
            raise ValueError('explicit separate endpoints required')
        canonical = _canonical(data)
        object.__setattr__(self, 'manifest_json', canonical)
        object.__setattr__(self, 'protocol_id', sha256(canonical.encode('utf-8')).hexdigest())

    @property
    def horizon_seconds(self):
        return json.loads(self.manifest_json)['horizon_seconds']


def load_protocol():
    """Explicit local manifest read; importing the module performs no file I/O."""
    return ProtocolDefinition(Path(__file__).with_name('protocol_v01.json').read_text(encoding='utf-8'))


@dataclass(frozen=True)
class TrialAnnotation(_Record):
    sequence_id: str
    split: str
    media_reference: str
    annotation_reference: str
    status: str
    actor_category: str
    region_id: str
    frame_timestamps: tuple
    visibility: tuple
    monitor_start_frame: int
    onset_frame: int | None = None
    observed_before_occlusion_frame: int | None = None
    occlusion_interval: tuple | None = None
    exclusion_reason: str | None = None

    def __post_init__(self):
        for name in ('sequence_id', 'split', 'media_reference', 'annotation_reference', 'actor_category', 'region_id'):
            _text(getattr(self, name))
        if self.status not in ('positive', 'negative', 'ambiguous', 'missing_annotation', 'censored'):
            raise ValueError('explicit annotation status required')
        for name in ('frame_timestamps', 'visibility'):
            values = getattr(self, name)
            if not isinstance(values, (tuple, list)):
                raise ValueError('ordered frame annotation required')
            object.__setattr__(self, name, tuple(values))
        n = len(self.frame_timestamps)
        if n < 2 or len(self.visibility) != n:
            raise ValueError('aligned frame timeline required')
        _index(self.monitor_start_frame, n)
        known = [t for t in self.frame_timestamps if t is not None]
        for t in known:
            _time(t)
        if any(a >= b for a, b in zip(known, known[1:])):
            raise ValueError('timestamps must strictly increase')
        if any(v not in ('absent', 'occluded', 'partial', 'visible', 'ambiguous', 'unannotated') for v in self.visibility):
            raise ValueError('unsupported visibility annotation')
        if self.observed_before_occlusion_frame is not None:
            _index(self.observed_before_occlusion_frame, n)
            if self.visibility[self.observed_before_occlusion_frame] != 'visible':
                raise ValueError('prior observation requires visible annotation')
        if self.occlusion_interval is not None:
            if not isinstance(self.occlusion_interval, (tuple, list)) or len(self.occlusion_interval) != 2:
                raise ValueError('inclusive occlusion interval required')
            start, end = self.occlusion_interval
            _index(start, n)
            _index(end, n)
            if start > end or any(v not in ('occluded', 'partial') for v in self.visibility[start:end + 1]):
                raise ValueError('invalid occlusion interval')
            if self.observed_before_occlusion_frame is not None and self.observed_before_occlusion_frame >= start:
                raise ValueError('prior observation must precede occlusion')
            object.__setattr__(self, 'occlusion_interval', (start, end))
        if self.status in ('positive', 'negative'):
            if self.exclusion_reason is not None or any(v in ('ambiguous', 'unannotated') for v in self.visibility[self.monitor_start_frame:]):
                raise ValueError('resolved annotation cannot hide ambiguity')
        else:
            _text(self.exclusion_reason)
        if self.status == 'positive':
            _index(self.onset_frame, n)
            visible = [i for i in range(self.monitor_start_frame, n) if self.visibility[i] == 'visible']
            if not visible or visible[0] != self.onset_frame or self.onset_frame + 1 >= n or self.visibility[self.onset_frame + 1] != 'visible':
                raise ValueError('onset must be first visible frame confirmed by next frame')
            if self.occlusion_interval is not None and self.occlusion_interval[1] >= self.onset_frame:
                raise ValueError('occlusion must precede onset')
        elif self.onset_frame is not None:
            raise ValueError('only resolved positive annotations have onset')
        if self.status == 'negative' and 'visible' in self.visibility[self.monitor_start_frame:]:
            raise ValueError('negative trial cannot contain target appearance')


@dataclass(frozen=True)
class PredictionWindow(_Record):
    prediction_id: str
    sequence_id: str
    condition: str
    cutoff_frame: int
    issued_timestamp: float | None
    evidence_through_timestamp: float | None
    horizon_seconds: float
    decision: str
    actor_category: str | None = None
    region_id: str | None = None
    source_references: tuple = ()

    def __post_init__(self):
        for value in (self.prediction_id, self.sequence_id):
            _text(value)
        if self.condition not in CONDITIONS or self.decision not in ('anticipate', 'abstain'):
            raise ValueError('known condition and explicit decision required')
        if type(self.cutoff_frame) is not int or self.cutoff_frame < 0:
            raise ValueError('nonnegative cutoff frame required')
        _time(self.horizon_seconds)
        if self.horizon_seconds <= 0:
            raise ValueError('positive horizon required')
        for value in (self.issued_timestamp, self.evidence_through_timestamp):
            if value is not None:
                _time(value)
        if self.issued_timestamp is not None and self.evidence_through_timestamp is not None and self.evidence_through_timestamp > self.issued_timestamp:
            raise ValueError('future evidence cannot precede issuance')
        if self.decision == 'anticipate':
            _text(self.actor_category)
            _text(self.region_id)
        elif self.actor_category is not None or self.region_id is not None:
            raise ValueError('abstention cannot carry a target commitment')
        if not isinstance(self.source_references, (tuple, list)) or not self.source_references:
            raise ValueError('explicit prefix/output lineage required')
        for ref in self.source_references:
            _text(ref)
        object.__setattr__(self, 'source_references', tuple(sorted(set(self.source_references))))


@dataclass(frozen=True)
class AnticipationMatch(_Record):
    prediction_id: str
    status: str
    reason: str
    lead_seconds: float | None = None

    def __post_init__(self):
        _text(self.prediction_id)
        _text(self.reason)
        if self.status not in ('valid', 'false_anticipation', 'abstained', 'at_or_after_onset', 'unavailable'):
            raise ValueError('unknown match status')
        if self.status == 'valid':
            _time(self.lead_seconds)
            if self.lead_seconds <= 0:
                raise ValueError('anticipation requires positive lead')
        elif self.lead_seconds is not None:
            raise ValueError('unavailable anticipation cannot carry TTA')


def match_anticipation(protocol, trial, prediction):
    """Apply fixed matching to supplied records; does not infer or produce predictions."""
    if type(protocol) is not ProtocolDefinition or type(trial) is not TrialAnnotation or type(prediction) is not PredictionWindow:
        raise ValueError('typed protocol, annotation and prediction required')
    p = prediction
    if p.sequence_id != trial.sequence_id or p.horizon_seconds != protocol.horizon_seconds:
        raise ValueError('sequence or horizon mismatch')
    _index(p.cutoff_frame, len(trial.frame_timestamps))
    if p.cutoff_frame < trial.monitor_start_frame:
        raise ValueError('prediction outside fixed monitoring interval')
    stamp = trial.frame_timestamps[p.cutoff_frame]
    if stamp is not None and p.issued_timestamp is not None and stamp != p.issued_timestamp:
        raise ValueError('issuance must equal source cutoff time; latency is separate')
    def result(status, reason, lead=None):
        return AnticipationMatch(p.prediction_id, status, reason, lead)
    if trial.status not in ('positive', 'negative'):
        return result('unavailable', trial.status)
    if any(t is None for t in trial.frame_timestamps) or p.issued_timestamp is None or p.evidence_through_timestamp is None:
        return result('unavailable', 'missing_timestamp')
    if p.evidence_through_timestamp != stamp:
        raise ValueError('stale or unmatched observation prefix')
    # Complete horizon is required for every condition, even if an event is known early.
    if stamp + p.horizon_seconds > trial.frame_timestamps[-1]:
        return result('unavailable', 'right_censored_window')
    if trial.status == 'positive':
        onset = trial.frame_timestamps[trial.onset_frame]
        if stamp >= onset:
            return result('at_or_after_onset', 'not_pre_event')
    if p.decision == 'abstain':
        return result('abstained', 'explicit_abstention')
    if trial.status == 'negative':
        return result('false_anticipation', 'resolved_no_target_event')
    if (p.actor_category, p.region_id) != (trial.actor_category, trial.region_id):
        return result('false_anticipation', 'target_mismatch')
    if onset > stamp + p.horizon_seconds:
        return result('false_anticipation', 'target_outside_issued_horizon')
    return result('valid', 'exact_category_region_and_time', onset - stamp)


@dataclass(frozen=True)
class TTAResult(_Record):
    protocol_id: str
    sequence_id: str
    condition: str
    status: str
    first_prediction_id: str | None
    value: float | None
    matches: tuple
    missing_cutoff_frames: tuple
    units: str = field(default='seconds', init=False)

    def __post_init__(self):
        for value in (self.protocol_id, self.sequence_id):
            _text(value)
        if self.condition not in CONDITIONS or self.status not in ('available', 'no_anticipation', 'negative_trial', 'annotation_unavailable', 'missing_timestamp', 'incomplete_prediction_log', 'no_eligible_windows'):
            raise ValueError('invalid TTA status')
        if not isinstance(self.matches, (tuple, list)) or any(type(m) is not AnticipationMatch for m in self.matches):
            raise ValueError('typed matches required')
        object.__setattr__(self, 'matches', tuple(self.matches))
        if not isinstance(self.missing_cutoff_frames, (tuple, list)) or any(type(i) is not int or i < 0 for i in self.missing_cutoff_frames):
            raise ValueError('explicit missing frames required')
        object.__setattr__(self, 'missing_cutoff_frames', tuple(sorted(set(self.missing_cutoff_frames))))
        if self.status == 'available':
            _time(self.value)
            if self.value <= 0 or self.missing_cutoff_frames or not any(m.status == 'valid' and m.prediction_id == self.first_prediction_id and m.lead_seconds == self.value for m in self.matches):
                raise ValueError('TTA requires contributing valid anticipation')
        elif self.value is not None or self.first_prediction_id is not None:
            raise ValueError('missing TTA must remain None')


def assess_trial(protocol, trial, predictions, *, condition):
    """Validate one condition's log; fixture use only during protocol development."""
    if type(protocol) is not ProtocolDefinition or type(trial) is not TrialAnnotation or condition not in CONDITIONS:
        raise ValueError('typed context and known condition required')
    if not isinstance(predictions, (tuple, list)) or any(type(p) is not PredictionWindow for p in predictions):
        raise ValueError('typed prediction log required')
    by_id, by_frame = {}, {}
    for p in predictions:
        if p.condition != condition:
            raise ValueError('mixed conditions')
        if p.prediction_id in by_id:
            if by_id[p.prediction_id] != p:
                raise ValueError('prediction identity rewritten')
            continue  # exact transport duplicates do not earn another vote
        if p.cutoff_frame in by_frame:
            raise ValueError('multiple or contradictory commitments at one cutoff')
        by_id[p.prediction_id], by_frame[p.cutoff_frame] = p, p
    ordered = tuple(sorted(by_id.values(), key=lambda p: (p.cutoff_frame, p.prediction_id)))
    matches = tuple(match_anticipation(protocol, trial, p) for p in ordered)
    missing = ()
    if trial.status not in ('positive', 'negative'):
        status = 'annotation_unavailable'
    elif any(t is None for t in trial.frame_timestamps):
        status = 'missing_timestamp'
    else:
        scheduled = tuple(i for i, t in enumerate(trial.frame_timestamps) if i >= trial.monitor_start_frame and t + protocol.horizon_seconds <= trial.frame_timestamps[-1])
        missing = tuple(i for i in scheduled if i not in by_frame)
        if not scheduled:
            status = 'no_eligible_windows'
        elif missing:
            status = 'incomplete_prediction_log'
        elif any(m.status == 'unavailable' and m.reason == 'missing_timestamp' for m in matches):
            status = 'missing_timestamp'
        elif trial.status == 'negative':
            status = 'negative_trial'
        else:
            status = 'available' if any(m.status == 'valid' for m in matches) else 'no_anticipation'
    first = next((m for m in matches if m.status == 'valid'), None) if status == 'available' else None
    return TTAResult(protocol.protocol_id, trial.sequence_id, condition, status,
        None if first is None else first.prediction_id, None if first is None else first.lead_seconds, matches, missing)


@dataclass(frozen=True)
class AcousticSlot(_Record):
    sequence_id: str
    split: str
    cutoff_frame: int
    cutoff_timestamp: float
    representation_reference: str

    def __post_init__(self):
        for v in (self.sequence_id, self.split, self.representation_reference):
            _text(v)
        if type(self.cutoff_frame) is not int or self.cutoff_frame < 0:
            raise ValueError('nonnegative cutoff required')
        _time(self.cutoff_timestamp)


def shuffle_assignments(slots):
    """Return (recipient, donor-or-None) pairs; never rewrite source representations."""
    if not isinstance(slots, (tuple, list)) or any(type(s) is not AcousticSlot for s in slots):
        raise ValueError('typed acoustic slots required')
    groups, unique, splits = {}, set(), {}
    for slot in slots:
        key = (slot.split, slot.cutoff_frame, slot.cutoff_timestamp)
        identity = (slot.split, slot.cutoff_frame, slot.sequence_id)
        if identity in unique:
            raise ValueError('duplicate shuffle slot')
        if slot.sequence_id in splits and splits[slot.sequence_id] != slot.split:
            raise ValueError('sequence cannot cross dataset split boundaries')
        splits[slot.sequence_id] = slot.split
        unique.add(identity)
        groups.setdefault(key, []).append(slot)
    pairs = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda s: s.sequence_id)
        pairs.extend((s, group[(i + 1) % len(group)] if len(group) > 1 else None) for i, s in enumerate(group))
    return tuple(pairs)
