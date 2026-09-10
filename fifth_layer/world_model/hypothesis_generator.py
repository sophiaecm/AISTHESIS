"""Deterministic, uncalibrated candidate rules. No memory or model dependency."""
from collections import defaultdict

from ._structured import number
from .evidence import stable_id
from .sensory_evidence import sensory_applies
from .hypothesis import Hypothesis, HypothesisSet
from .trajectory import FutureTrajectory, validate_trajectories


STATEMENTS = {
    'continued_motion': 'The associated object may continue its reported motion.',
    'object_stops': 'The associated object may stop or remain stationary.',
    'object_becomes_occluded': 'The associated object may become or remain partly occluded.',
    'object_reappears': 'The previously associated object may become visible again.',
    'unknown_dynamic_event': 'Available evidence is insufficient to characterize a future event.',
}


def _association(item):
    if item.track_id is not None:
        return ('track', type(item.track_id).__name__, item.track_id)
    if item.object_id is not None:
        return ('object', type(item.object_id).__name__, item.object_id)
    # Unbound evidence cannot silently refer to the same object across providers.
    return ('unbound', item.evidence_id)


class MultiHypothesisGenerator:
    """Generate active alternatives, never an automatic winner.

    Direct rule score=1; moving-object stop alternative=.25; unknown=0.
    Any explicit opposing evidence halves the score once, without deleting the
    candidate. posterior_probability stores this UNCALIBRATED compatibility
    score; HypothesisSet alone supplies normalized ranking weights. Prior=.5
    is a neutral heuristic placeholder, not an empirical prior. Confidence is
    copied from one deterministic supporting source, or 0 with assessed=False.
    It is never averaged or used to calculate the ranking score.
    """
    def generate(self, scene, evidence, *, horizon_seconds=1.0, created_timestamp=None):
        if evidence.scene_id != scene.scene_id:
            raise ValueError('EvidenceBundle.scene_id must match SceneState.scene_id')
        created = scene.timestamp if created_timestamp is None else created_timestamp
        if created is None:
            raise ValueError('created_timestamp is required when SceneState.timestamp is unknown')
        number(created, 'created_timestamp', nonnegative=True)
        number(horizon_seconds, 'horizon_seconds', nonnegative=True)
        groups = defaultdict(list)
        sensory = defaultdict(list)
        for item in evidence.items:
            if item.source_type == 'experience':
                # Historical context cannot generate candidates or change base scores.
                continue
            if item.source_type == 'sensory':
                if item.epistemic_status != 'unavailable' and (item.track_id is not None or item.object_id is not None):
                    sensory[_association(item)].append(item)
                continue
            groups[_association(item)].append(item)
        if not groups:
            groups[('scene', scene.scene_id)] = []
        hypotheses = []
        for association in sorted(groups, key=lambda key: stable_id('association', key)):
            items = sorted(groups[association], key=lambda item: item.evidence_id)
            candidates = {}
            for kind in STATEMENTS:
                if kind == 'unknown_dynamic_event':
                    continue
                supporting = [item for item in items if kind in item.supports]
                if supporting:
                    candidates[kind] = (supporting, 'explicit_evidence', 1.0,
                                        ('Source evidence is a fallible indication of a possible future.',))
            moving = [item for item in items if 'continued_motion' in item.supports]
            occluded = [item for item in items if 'object_becomes_occluded' in item.supports]
            if moving and 'object_stops' not in candidates:
                candidates['object_stops'] = (moving, 'motion_stop_alternative', .25,
                    ('Reported motion may change within the horizon; stopping is an alternative, not an observation.',))
            if moving and occluded and association[0] in ('track', 'object'):
                candidates.setdefault('object_reappears', (list({x.evidence_id: x for x in moving + occluded}.values()),
                    'occluded_motion_continuity', 1.0,
                    ('Motion and occlusion refer to the same explicit source identity.',
                     'The existing object may persist while hidden; reappearance is not guaranteed.')))
            if not candidates:
                candidates['unknown_dynamic_event'] = (items, 'insufficient_evidence', 0.0,
                    ('Missing or unbound evidence does not establish a negative outcome.',))
            for kind in sorted(candidates):
                supporting, rule, raw, assumptions = candidates[kind]
                opposing = [item for item in items if kind in item.contradicts]
                additional = [item for item in sensory[association]
                              if sensory_applies(item, kind, items)]
                sensory_opposing = [item for item in sensory[association]
                                    if sensory_applies(item, kind, items, opposing=True)]
                opposing = sorted(opposing + sensory_opposing, key=lambda item: item.evidence_id)
                score = raw * (.5 if opposing else 1.)
                supporting = sorted(supporting, key=lambda item: item.evidence_id)
                confidence_source = next((item for item in supporting if item.confidence is not None), None)
                # Correlated inferred consequences never raise base confidence or score.
                supporting = sorted(supporting + additional, key=lambda item: item.evidence_id)
                identity = stable_id('h', scene.scene_id, association, kind, created, horizon_seconds)
                hypotheses.append(Hypothesis(
                    hypothesis_id=identity, scene_id=scene.scene_id, hypothesis_type=kind,
                    statement=STATEMENTS[kind], prior_probability=.5, posterior_probability=score,
                    confidence=confidence_source.confidence if confidence_source else 0.,
                    created_timestamp=created, horizon_seconds=horizon_seconds,
                    target_timestamp=created + horizon_seconds,
                    track_id=association[2] if association[0] == 'track' else None,
                    evidence_for=tuple(item.evidence_id for item in supporting),
                    evidence_against=tuple(item.evidence_id for item in opposing),
                    assumptions=assumptions,
                    provenance=dict(generator_version='0.2', generation_rule=rule,
                        association=association, score_policy='heuristic_v0.2', calibrated=False,
                        base_score=raw, contradiction_multiplier=.5 if opposing else 1., raw_score=score,
                        prior_policy='constant neutral heuristic 0.5; not empirical',
                        confidence_assessed=confidence_source is not None,
                        confidence_evidence_id=confidence_source.evidence_id if confidence_source else None,
                        score_inputs=tuple(dict(evidence_id=item.evidence_id, confidence=item.confidence,
                                                supports=item.supports, contradicts=item.contradicts,
                                                value=item.value, **(dict(
                                                    epistemic_status=item.epistemic_status,
                                                    modality=item.modality,
                                                    supporting_evidence_ids=item.supporting_evidence_ids,
                                                    opposing_evidence_ids=item.opposing_evidence_ids,
                                                    provenance=item.provenance)
                                                    if item.source_type == 'sensory' else {}))
                                           for item in sorted(items + additional + sensory_opposing,
                                                              key=lambda item: item.evidence_id)),
                        scene_uncertainty=scene.uncertainty)))
        return HypothesisSet(scene.scene_id, tuple(hypotheses), created)


def adapt_temporal_trajectories(scene, evidence, hypotheses):
    """Link supplied temporal centers to continuation hypotheses only.

    Preserve points and source timestamps. Never extrapolate, invent bboxes,
    convert pixel uncertainty, or attach motion paths to stopping alternatives.
    Out-of-horizon points and sources from a different issuance time are not
    linked; empty/incompatible sources yield no trajectory.
    """
    if scene.scene_id != evidence.scene_id or scene.scene_id != hypotheses.scene_id:
        raise ValueError('scene_id must match evidence and hypotheses')
    trajectories = []
    for hypothesis in hypotheses.hypotheses:
        if hypothesis.hypothesis_type != 'continued_motion':
            continue
        for item in evidence.items:
            if (item.source_type != 'temporal' or item.evidence_id not in hypothesis.evidence_for
                    or item.timestamp != hypothesis.created_timestamp):
                continue
            points = item.value.get('trajectory', ())
            if not points or item.value.get('trajectory_available') is False:
                continue
            compatible = []
            for point in points:
                if 'horizon_seconds' not in point or 'center' not in point:
                    continue
                number(point['horizon_seconds'], 'trajectory.horizon_seconds', nonnegative=True)
                if point['horizon_seconds'] <= hypothesis.horizon_seconds:
                    compatible.append(point)
            if not compatible:
                continue
            trajectories.append(FutureTrajectory(
                trajectory_id=stable_id('trajectory', hypothesis.hypothesis_id, item.evidence_id),
                hypothesis_id=hypothesis.hypothesis_id, track_id=hypothesis.track_id,
                start_timestamp=item.timestamp,
                horizon_seconds=max(point['horizon_seconds'] for point in compatible),
                predicted_states=tuple(compatible),
                predicted_centers=tuple(point['center'] for point in compatible),
                expected_event=hypothesis.hypothesis_type,
                provenance=dict(source_evidence_id=item.evidence_id, source_component=item.source_component,
                                source_field='trajectory', source_timestamp=item.timestamp,
                                source_provenance=item.provenance, adapter_version='0.2')))
    return validate_trajectories(trajectories, hypotheses)
