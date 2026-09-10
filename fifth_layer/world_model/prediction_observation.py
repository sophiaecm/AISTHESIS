"""Deterministic prediction/outcome comparison with no inference feedback."""
from collections.abc import Mapping
from math import hypot

from ._structured import freeze, number
from .evidence import stable_id
from .experience_memory import ExperienceEpisode
from .prediction_records import PredictionRecord, OutcomeRecord, PredictionEvaluation


MOVING = frozenset(('moving', 'moving_left', 'moving_right', 'moving_up', 'moving_down'))
MOTION = MOVING | {'stationary'}


def _same(a, b):
    return a is not None and type(a) is type(b) and a == b


def _size(scene):
    if scene.image_width and scene.image_height:
        return (scene.image_width, scene.image_height)
    return None


def _unique(values):
    values = {stable_id('value', value): value for value in values if value is not None}
    return next(iter(values.values())) if len(values) == 1 else None


def prediction_from_hypothesis(scene, hypothesis, evidence, trajectory=None):
    """Project an issued hypothesis, never regenerate or strengthen its claims.

    Only a trajectory point explicitly timed at the hypothesis target is used.
    No interpolation, extrapolation, fabricated boxes, or direction from class.
    """
    h = hypothesis
    if scene.scene_id != h.scene_id or scene.scene_id != evidence.scene_id:
        raise ValueError('prediction sources must share scene_id')
    state = {}
    if h.hypothesis_type != 'unknown_dynamic_event':
        state['event'] = h.hypothesis_type
    association = h.provenance.get('association', ())
    object_id = association[2] if len(association) == 3 and association[0] == 'object' else None
    supporting = [x for x in evidence.items if x.evidence_id in h.evidence_for]
    if h.hypothesis_type == 'continued_motion':
        motion = _unique(x.value.get('motion_state') for x in supporting
                         if x.source_type in ('motion', 'temporal')
                         and x.timestamp == h.created_timestamp
                         and (_same(x.track_id, h.track_id) if h.track_id is not None
                              else _same(x.object_id, object_id)))
        if motion in MOVING:
            state['motion_state'] = motion
    elif h.hypothesis_type == 'object_stops':
        state['motion_state'] = 'stationary'
    elif h.hypothesis_type == 'object_becomes_occluded':
        state['visibility'] = 'occluded'
    elif h.hypothesis_type == 'object_reappears':
        state['visibility'] = 'visible'
    trajectory_id = None
    if trajectory is not None:
        if (trajectory.hypothesis_id != h.hypothesis_id
                or ((trajectory.track_id is not None or h.track_id is not None)
                    and not _same(trajectory.track_id, h.track_id))
                or trajectory.start_timestamp != h.created_timestamp
                or trajectory.expected_event not in (None, h.hypothesis_type)):
            raise ValueError('trajectory does not match prediction identity/time')
        trajectory_id = trajectory.trajectory_id
        centers, boxes = [], []
        for index, point in enumerate(trajectory.predicted_states):
            if point.get('horizon_seconds') == h.horizon_seconds <= trajectory.horizon_seconds:
                center = point.get('center')
                if center is None and trajectory.predicted_centers:
                    center = trajectory.predicted_centers[index]
                centers.append(center)
                box = point.get('bbox')
                if box is None and trajectory.predicted_bboxes:
                    box = trajectory.predicted_bboxes[index]
                boxes.append(box)
        center = _unique(centers)
        if center is not None:
            state['center'] = center
        box = _unique(boxes)
        if box is not None:
            state['bbox'] = box
    values = dict(scene_id=scene.scene_id, hypothesis_id=h.hypothesis_id,
        hypothesis_type=h.hypothesis_type, source_timestamp=h.created_timestamp,
        target_timestamp=h.target_timestamp, horizon_seconds=h.horizon_seconds,
        predicted_state=state, track_id=h.track_id, object_id=object_id,
        trajectory_id=trajectory_id, image_size=_size(scene),
        confidence=None if h.provenance.get('confidence_assessed') is False else h.confidence,
        uncertainty=scene.uncertainty, evidence_for=tuple(sorted(h.evidence_for)),
        evidence_against=tuple(sorted(h.evidence_against)),
        provenance=dict(adapter_version='experience-0.1', hypothesis_provenance=h.provenance,
            source_scene_provenance=scene.provenance,
            trajectory_provenance=trajectory.provenance if trajectory else {},
            source_statement=h.statement, source_assumptions=h.assumptions,
            source_prior=h.prior_probability, source_posterior=h.posterior_probability,
            geometry_policy='only an explicitly timed target point; no interpolation',
            confidence_policy='source confidence only; unassessed is unknown'))
    return PredictionRecord(prediction_id=stable_id('prediction', values), **values)


def outcome_from_scene(prediction, scene, *, sensory_evidence=None, object_ids_persistent=False):
    """Read later observed data. Prediction content is never an outcome source.

    Track IDs are scoped to a caller's continuous tracking session. Object IDs
    are frame-local by default and cannot establish cross-frame identity.
    Optional explicit observation annotations in occlusion_evidence require
    epistemic_status='observed'; raw overlap/truncation is never true occlusion.
    """
    p = prediction
    field = 'track_id' if p.track_id is not None else 'object_id'
    identity = p.track_id if field == 'track_id' else p.object_id
    status, observed, unavailable, sources, association_candidates = 'missing', {}, [], [], []
    associated_confidence = None
    if identity is None or (field == 'object_id' and not object_ids_persistent):
        status = 'unbound'
        unavailable.append('insufficient_association_evidence')
    elif scene is not None:
        def matches(record):
            return _same(record.get(field), identity)

        def fresh(record):
            return (not record.get('is_predicted')
                    and record.get('observed', True) is True
                    and record.get('observation_state', 'observed') == 'observed'
                    and record.get('epistemic_status', 'observed') == 'observed'
                    and record.get('observation_timestamp', record.get('timestamp', scene.timestamp)) == scene.timestamp)

        objects = [x for x in scene.observed_objects if matches(x) and fresh(x)]
        occlusion = scene.occlusion_evidence
        occlusion = occlusion.get('occlusion_evidence', ()) if isinstance(occlusion, Mapping) else occlusion
        explicit = [x for x in occlusion
                    if matches(x) and fresh(x) and x.get('epistemic_status') == 'observed']
        association_candidates = sorted(objects + explicit, key=lambda x: stable_id('association-candidate', x))
        associated_confidence = _unique(x.get('association_confidence') for x in association_candidates)
        if len(objects) > 1 or any(x.get('association_reliable') is False for x in objects + explicit):
            status = 'ambiguous'
            unavailable.append('ambiguous_association')
        elif objects or explicit:
            status = 'matched'
            motions = [x for x in scene.motion_evidence if matches(x) and fresh(x)]
            records = objects + explicit + motions
            for name in ('motion_state', 'visibility', 'event_timestamp'):
                value = _unique(x.get(name) for x in records)
                if value is not None:
                    observed[name] = value
                elif any(x.get(name) is not None for x in records):
                    unavailable.append(f'conflicting_{name}')
            # Only actual detector geometry, never predicted_tracks or future data.
            if objects:
                obj = objects[0]
                center = obj.get('center')
                if center is None and obj.get('box_xyxy') is not None:
                    x1, y1, x2, y2 = obj['box_xyxy']
                    center = ((x1 + x2) / 2., (y1 + y2) / 2.)
                elif center is None and obj.get('box') is not None:
                    x, y, width, height = obj['box']
                    center = (x + width / 2., y + height / 2.)
                if center is not None:
                    observed['center'] = center
            # Explicit event booleans only; absence of an event key is unknown.
            event_names = sorted({name for x in records for name in x.get('events', {})})
            events = {}
            for name in event_names:
                value = _unique(x.get('events', {}).get(name) for x in records)
                if value is not None:
                    events[name] = value
                else:
                    unavailable.append(f'conflicting_event:{name}')
            if events:
                observed['events'] = events
            sources = sorted((dict(source='SceneState.observed_objects', record=x) for x in objects),
                             key=lambda x: stable_id('source', x))
            sources += sorted((dict(source='SceneState.motion_evidence', record=x) for x in motions),
                              key=lambda x: stable_id('source', x))
            sources += sorted((dict(source='SceneState.occlusion_evidence.explicit_observation', record=x)
                               for x in explicit), key=lambda x: stable_id('source', x))
        else:
            unavailable.append('track_not_observed; loss_is_not_occlusion')
    else:
        unavailable.append('no_later_scene')
    if sensory_evidence is not None and (scene is None or sensory_evidence.scene_id != scene.scene_id):
        raise ValueError('sensory outcome evidence must match later scene')
    if p.predicted_state.get('modality'):
        candidates = [] if sensory_evidence is None else [x for x in sensory_evidence.items
            if x.source_type == 'sensory' and x.epistemic_status == 'observed'
            and x.modality == p.predicted_state['modality'] and _same(getattr(x, field), identity)
            and x.timestamp == scene.timestamp]
        claim = _unique(x.value.get('claim') for x in candidates)
        if status == 'matched' and claim is not None:
            observed['sensory_claim'] = claim
            sources += [dict(source='sensory_evidence', evidence_id=x.evidence_id, provenance=x.provenance)
                        for x in candidates]
        else:
            unavailable.append('sensory_observation_unavailable')
    values = dict(prediction_id=p.prediction_id, scene_id=scene.scene_id if scene else None,
        observation_timestamp=scene.timestamp if scene else None, association_status=status,
        observed_state=observed, track_id=p.track_id if status == 'matched' else None,
        object_id=p.object_id if status == 'matched' else None, image_size=_size(scene) if scene else None,
        association_confidence=associated_confidence, unavailable=tuple(sorted(set(unavailable))),
        provenance=dict(adapter_version='experience-0.1', association_policy='typed exact ID; no class/nearest rematch',
            object_ids_persistent=object_ids_persistent, sources=sources,
            association_candidates=association_candidates,
            scene_provenance=scene.provenance if scene else {}))
    return OutcomeRecord(outcome_id=stable_id('outcome', values), **values)


def evaluate_prediction(prediction, outcome, *, time_tolerance_seconds=0., position_tolerance_pixels=0.):
    """Compare target-time state, not the truth of a probabilistic possibility.

    Exact target time/position are defaults, explicitly recorded comparison
    policy. Tolerances do not modify any inference threshold. No event onset is
    inferred from a snapshot timestamp. Missing measurements remain None.
    """
    p, o = prediction, outcome
    if p.prediction_id != o.prediction_id:
        raise ValueError('outcome must refer to prediction')
    number(time_tolerance_seconds, 'time_tolerance_seconds', nonnegative=True)
    number(position_tolerance_pixels, 'position_tolerance_pixels', nonnegative=True)
    metrics = dict(event_match=None, motion_state_match=None, position_error_pixels=None,
        normalized_position_error=None, position_match=None, visibility_match=None,
        temporal_error_seconds=None, sensory_match=None, association_confidence=o.association_confidence)
    reasons, missing = list(o.unavailable), []
    status = None
    stamp = o.observation_timestamp
    if o.association_status != 'matched':
        status = 'unobservable' if o.association_status == 'missing' else 'insufficient_evidence'
        reasons.append('association_failure')
    elif not (_same(p.track_id, o.track_id) if p.track_id is not None else _same(p.object_id, o.object_id)):
        status = 'insufficient_evidence'
        reasons.append('association_identity_mismatch')
    elif stamp is None or stamp <= p.source_timestamp or abs(stamp - p.target_timestamp) > time_tolerance_seconds + 1e-9:
        status = 'insufficient_evidence'
        reasons.append('no_observation_at_target_time')
    if status is None:
        predicted, actual = p.predicted_state, o.observed_state
        matches = []

        def compare(name, value):
            metrics[name] = value
            if value is None:
                missing.append(name)
            else:
                matches.append(value)

        if predicted.get('motion_state') is not None:
            expected, observed = predicted['motion_state'], actual.get('motion_state')
            match = None
            if expected in MOTION and observed in MOTION:
                match = observed in MOVING if expected == 'moving' else (
                    None if observed == 'moving' and expected in MOVING else expected == observed)
            compare('motion_state_match', match)
        if predicted.get('visibility') is not None:
            compare('visibility_match', None if actual.get('visibility') is None
                    else predicted['visibility'] == actual['visibility'])
        event = predicted.get('event')
        if event is not None:
            match = actual.get('events', {}).get(event)
            if match is None and not any(x == f'conflicting_event:{event}' for x in o.unavailable):
                if event in ('continued_motion', 'object_stops'):
                    match = metrics['motion_state_match']
                elif event in ('object_becomes_occluded', 'object_reappears'):
                    match = metrics['visibility_match']
            compare('event_match', match)
        if predicted.get('center') is not None:
            if actual.get('center') is None:
                compare('position_match', None)
            elif p.image_size is None or o.image_size != p.image_size:
                compare('position_match', None)
                reasons.append('unknown_or_changed_image_geometry')
            else:
                error = hypot(*(a - b for a, b in zip(predicted['center'], actual['center'])))
                metrics['position_error_pixels'] = error
                metrics['normalized_position_error'] = error / hypot(*p.image_size)
                compare('position_match', error <= position_tolerance_pixels)
        if predicted.get('sensory_claim') is not None:
            compare('sensory_match', None if actual.get('sensory_claim') is None else
                    predicted['sensory_claim'] == actual['sensory_claim'])
        if (event is not None and actual.get('events', {}).get(event) is True
                and predicted.get('event_timestamp') is not None and actual.get('event_timestamp') is not None):
            metrics['temporal_error_seconds'] = actual['event_timestamp'] - predicted['event_timestamp']
        if not matches:
            status = 'unobservable' if 'sensory_observation_unavailable' in reasons else 'insufficient_evidence'
        elif all(matches) and not missing:
            status = 'supported'
        elif any(matches):
            status = 'partially_supported'
        else:
            status = 'contradicted'
    values = dict(prediction_id=p.prediction_id, outcome_id=o.outcome_id, status=status,
        evaluated_timestamp=stamp, metrics=metrics, reasons=tuple(sorted(set(reasons))),
        missing_fields=tuple(sorted(set(missing))), provenance=dict(evaluator_version='experience-0.1',
            time_tolerance_seconds=time_tolerance_seconds, position_tolerance_pixels=position_tolerance_pixels,
            normalization='image diagonal; same known image size required',
            interpretation='target-state compatibility; not probability calibration or causal falsification',
            learning=False))
    return PredictionEvaluation(evaluation_id=stable_id('evaluation', freeze(p), freeze(o), values), **values)


def experience_episode(prediction, outcome, evaluation):
    """Connect the new immutable records through existing summary slots."""
    if prediction.prediction_id != outcome.prediction_id or prediction.prediction_id != evaluation.prediction_id:
        raise ValueError('experience prediction IDs must match')
    if outcome.outcome_id != evaluation.outcome_id:
        raise ValueError('experience outcome IDs must match')
    return ExperienceEpisode(
        episode_id=stable_id('episode', prediction.prediction_id, outcome.outcome_id, evaluation.evaluation_id),
        source_scene_id=prediction.scene_id, hypothesis_id=prediction.hypothesis_id,
        created_timestamp=prediction.source_timestamp, target_scene_id=outcome.scene_id,
        trajectory_id=prediction.trajectory_id, prediction_id=prediction.prediction_id,
        prediction_summary=freeze(prediction), observation_summary=freeze(outcome),
        prediction_timestamp=prediction.source_timestamp, observation_timestamp=outcome.observation_timestamp,
        evaluation_status=evaluation.status, prediction_error=freeze(evaluation),
        evaluated_timestamp=evaluation.evaluated_timestamp,
        provenance=dict(recorder_version='experience-0.1', learning=False, inference_feedback=False))
