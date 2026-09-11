"""Model-agnostic, read-only constraints on observed image-plane snapshots."""
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from math import hypot

from ._structured import geometry, number
from .evidence import stable_id
from .physical_state import PhysicalWorldState, plain
from .physics_constraints import PhysicsConstraintResult, PhysicsConstraintBundle, PhysicsTransitionAssessment


@dataclass(frozen=True)
class PhysicsConstraintPolicy:
    # None disables heuristic judgement, not measurement reporting.
    max_displacement_pixels: float | None = None
    max_velocity_change_pixels_per_second: float | None = None
    support_gap_pixels: float = 0.

    def __post_init__(self):
        for name in ('max_displacement_pixels', 'max_velocity_change_pixels_per_second', 'support_gap_pixels'):
            value = getattr(self, name)
            if value is not None:
                number(value, name, nonnegative=True)
        if self.support_gap_pixels is None:
            raise ValueError('support_gap_pixels requires an explicit nonnegative value')


def _ref(state, obj=None, field=None):
    base = f'{state.scene_id}@{state.timestamp!r}'
    if obj is not None:
        base += f'.objects[{obj.object_id!r}]'
    return base + ('.' + field if field else '')


def _attribute(obj, name, *, units=None, possible=False):
    attr = obj.attributes.get(name)
    if attr is None or not attr.derived_from or attr.value is None:
        return None
    if attr.status not in (('observed', 'estimated', 'possible') if possible else ('observed', 'estimated')):
        return None
    if units is not None and attr.units != units:
        return None
    return attr.value


def _center(obj):
    if _attribute(obj, 'visibility_state') != 'observed':
        return None
    value = _attribute(obj, 'center', units='pixels')
    if value is not None:
        geometry(value, 'center', 2)
    return value


def _bbox(obj):
    if _attribute(obj, 'visibility_state') != 'observed':
        return None
    value = _attribute(obj, 'bbox', units='pixels')
    if value is not None:
        geometry(value, 'bbox', 4)
        if value[2] == value[0] or value[3] == value[1]:
            return None
    return value


def _tracks(state):
    result = {}
    for obj in state.objects:
        track = _attribute(obj, 'track_id')
        if track is None or _attribute(obj, 'visibility_state') != 'observed':
            continue
        if type(track) not in (int, str):
            raise ValueError('track identity must be integer or string')
        key = (type(track).__name__, track)
        if key in result:
            raise ValueError('ambiguous duplicate track in physical snapshot')
        result[key] = obj
    return result


def _touch(a, b):
    return min(a[2], b[2]) >= max(a[0], b[0]) and min(a[3], b[3]) >= max(a[1], b[1])


def _check_times(value, cutoff):
    if isinstance(value, Mapping):
        for key, child in value.items():
            if (key == 'timestamp' or key.endswith('_timestamp')) and child is not None:
                number(child, key, nonnegative=True)
                if cutoff is None or child > cutoff:
                    raise ValueError('future or unorderable physical source timestamp')
            elif isinstance(child, (Mapping, tuple, list)):
                _check_times(child, cutoff)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _check_times(child, cutoff)


class PhysicsConstraintEngine:
    """Caller supplies adjacent snapshots, in one session and fixed pixel frame.

    History is chronological, excludes current, and may include absences. Unknown
    timestamps allow ordered geometry comparisons but never velocity or inertia.
    This API evaluates supplied observations, not arbitrary hypothesis text.
    """
    def __init__(self, policy=None):
        self.policy = policy or PhysicsConstraintPolicy()
        if not isinstance(self.policy, PhysicsConstraintPolicy):
            raise ValueError('policy must be PhysicsConstraintPolicy')

    def assess(self, current, *, history=(), session_id, coordinate_frame_id):
        from ._structured import identifier
        identifier(session_id, 'session_id')
        identifier(coordinate_frame_id, 'coordinate_frame_id')
        states = tuple(history) + (current,)
        if any(not isinstance(s, PhysicalWorldState) or s.schema_version != 'physical-world-0.1' for s in states):
            raise ValueError('expected PhysicalWorldState v0.1 snapshots')
        if len({s.scene_id for s in states}) != len(states):
            raise ValueError('duplicate scene in transition history')
        known = [s.timestamp for s in states if s.timestamp is not None]
        if any(a >= b for a, b in zip(known, known[1:])):
            raise ValueError('history must strictly precede current and be chronological')
        if current.timestamp is None and any(s.timestamp is not None for s in history):
            raise ValueError('cannot order dated history against an undated current snapshot')
        for state in states:
            _check_times(state.provenance, state.timestamp)
            for obj in state.objects:
                _check_times(obj.provenance, state.timestamp)
                _center(obj)
                _bbox(obj)
        tracks = [_tracks(s) for s in states]
        now = tracks[-1]
        before = tracks[-2] if len(states) > 1 else {}
        prev = states[-2] if len(states) > 1 else None
        results = []
        policy = plain(self.policy)

        def emit(family, objects, status, finding, explanation, refs, measured=None, expected=None, uncertainty=()):
            values = dict(constraint_type=family, scene_id=current.scene_id, timestamp=current.timestamp,
                object_ids=tuple(sorted(set(objects))), status=status, finding=finding, explanation=explanation,
                measured_values=measured or {}, expected_range=expected or {}, derived_from=tuple(sorted(set(refs))),
                provenance={'session_id': session_id, 'coordinate_frame_id': coordinate_frame_id,
                    'policy': policy, 'epistemic_role': 'constraint_assessment_not_observed_fact'},
                uncertainty=tuple(sorted(set(uncertainty))), severity='review' if status == 'violated' else 'info')
            results.append(PhysicsConstraintResult(stable_id('constraint', values), **values))

        def refs(state, obj, *names):
            result = [_ref(state, obj, 'attributes.' + name) for name in names]
            for name in names:
                a = obj.attributes.get(name)
                if a is not None:
                    result.extend(a.derived_from)
            return result

        def transition(s0, o0, s1, o1):
            c0, c1 = _center(o0), _center(o1)
            dt = s1.timestamp-s0.timestamp if s0.timestamp is not None and s1.timestamp is not None else None
            delta = tuple(b-a for a, b in zip(c0, c1)) if c0 is not None and c1 is not None else None
            velocity = tuple(d/dt for d in delta) if delta is not None and dt is not None and dt > 0 else None
            return {'displacement_pixels': delta, 'distance_pixels': hypot(*delta) if delta is not None else None,
                    'temporal_gap_seconds': dt, 'velocity_pixels_per_second': velocity}

        for obj in current.objects:
            track = _attribute(obj, 'track_id')
            key = (type(track).__name__, track)
            old = before.get(key) if key in now else None
            source_refs = refs(current, obj, 'center', 'track_id', 'visibility_state') + [_ref(current, field='timestamp')]
            if old is not None:
                source_refs += refs(prev, old, 'center', 'track_id', 'visibility_state') + [_ref(prev, field='timestamp')]
            measurement = transition(prev, old, current, obj) if old else {
                'displacement_pixels': None, 'distance_pixels': None, 'temporal_gap_seconds': None, 'velocity_pixels_per_second': None}
            available = measurement['displacement_pixels'] is not None
            timed = measurement['velocity_pixels_per_second'] is not None
            emit('kinematic_continuity', (obj.object_id,), 'satisfied' if timed else 'indeterminate',
                'sampled_motion_continuity' if timed else 'insufficient_consecutive_observations',
                'Same-track center difference with explicit pixel units; does not establish continuous motion between samples.',
                source_refs, measurement, uncertainty=('sampling_gaps_and_camera_motion_unresolved',))
            limit = self.policy.max_displacement_pixels
            outlier = available and limit is not None and measurement['distance_pixels'] > limit
            emit('implausible_displacement', (obj.object_id,),
                'indeterminate' if not available else 'unsupported' if limit is None else 'violated' if outlier else 'satisfied',
                'kinematic_outlier' if outlier else 'no_outlier_under_configured_policy' if available and limit is not None else 'insufficient_geometry_or_threshold',
                'Pixel jump heuristic may indicate tracking discontinuity or jitter; not physical impossibility.',
                source_refs, measurement, {'max_displacement_pixels': limit, 'threshold_kind': 'heuristic'},
                ('no_metric_calibration',))
            inertia_refs = list(source_refs)
            change = None
            if old is not None and len(states) >= 3 and key in tracks[-3]:
                older = tracks[-3][key]
                first = transition(states[-3], older, prev, old)
                inertia_refs += refs(states[-3], older, 'center', 'track_id', 'visibility_state') + [_ref(states[-3], field='timestamp')]
                if first['velocity_pixels_per_second'] is not None and timed:
                    change = hypot(*(b-a for a, b in zip(first['velocity_pixels_per_second'], measurement['velocity_pixels_per_second'])))
            limit = self.policy.max_velocity_change_pixels_per_second
            abrupt = change is not None and limit is not None and change > limit
            emit('inertia_consistency', (obj.object_id,),
                'indeterminate' if change is None else 'unsupported' if limit is None else 'violated' if abrupt else 'satisfied',
                'abrupt_change_detected' if abrupt else 'motion_continuity_consistent' if change is not None and limit is not None else 'insufficient_history_or_threshold',
                'Compare consecutive sampled pixel-velocity vectors; no force or event cause is inferred.',
                inertia_refs, {'velocity_change_pixels_per_second': change},
                {'max_velocity_change_pixels_per_second': limit, 'threshold_kind': 'heuristic'},
                ('camera_motion_perspective_and_tracking_jitter_unresolved',))

        # Geometry predicates are possibilities; negative predicates are not negative physical facts.
        for a, b in combinations(current.objects, 2):
            ab, bb = _bbox(a), _bbox(b)
            pair_refs = refs(current, a, 'bbox', 'visibility_state') + refs(current, b, 'bbox', 'visibility_state')
            for upper, lower, ub, lb in ((a, b, ab, bb), (b, a, bb, ab)):
                supported = None if ub is None or lb is None else (
                    (ub[1]+ub[3])/2 < (lb[1]+lb[3])/2 and min(ub[2], lb[2]) > max(ub[0], lb[0])
                    and lb[1]-self.policy.support_gap_pixels <= ub[3] <= lb[3])
                emit('support_stability_possibility', (a.object_id, b.object_id),
                    'indeterminate' if supported is None else 'satisfied' if supported else 'not_applicable',
                    'indeterminate' if supported is None else 'support_geometry_possible' if supported else 'support_geometry_not_present',
                    'Upper bbox bottom meets lower vertical extent with horizontal overlap; geometry possibility only.',
                    pair_refs, {'upper_object_id': upper.object_id, 'lower_object_id': lower.object_id,
                                'upper_bbox_pixels': ub, 'lower_bbox_pixels': lb},
                    {'support_gap_pixels': self.policy.support_gap_pixels}, ('depth_contact_and_mechanical_stability_unknown',))
            approaching = None
            collision_refs = list(pair_refs)
            ta, tb = _attribute(a, 'track_id'), _attribute(b, 'track_id')
            oa, ob = before.get((type(ta).__name__, ta)), before.get((type(tb).__name__, tb))
            if oa is not None and ob is not None and prev.timestamp is not None and current.timestamp is not None:
                centers = [_center(o) for o in (oa, ob, a, b)]
                collision_refs += refs(prev, oa, 'center', 'track_id', 'visibility_state') + refs(prev, ob, 'center', 'track_id', 'visibility_state')
                collision_refs += refs(current, a, 'center', 'track_id') + refs(current, b, 'center', 'track_id')
                collision_refs += [_ref(prev, field='timestamp'), _ref(current, field='timestamp')]
                if all(c is not None for c in centers):
                    approaching = hypot(*(x-y for x, y in zip(centers[2], centers[3]))) < hypot(*(x-y for x, y in zip(centers[0], centers[1])))
            touching = None if ab is None or bb is None else _touch(ab, bb)
            finding = 'collision_possible' if touching and approaching else 'collision_geometry_absent' if touching is False else 'collision_indeterminate'
            emit('collision_possibility', (a.object_id, b.object_id),
                'satisfied' if finding == 'collision_possible' else 'not_applicable' if touching is False else 'indeterminate',
                finding, 'Approach plus current bbox contact/overlap is only a 2D collision possibility.',
                collision_refs, {'bbox_contact_or_overlap': touching, 'centers_approaching': approaching},
                uncertainty=('depth_and_actual_contact_unknown', 'events_between_snapshots_unknown'))

        # Missing objects are referenced at their earlier observed scene, never instantiated now.
        for key in sorted(set(before) | set(now), key=repr):
            obj, old = now.get(key), before.get(key)
            selected, source = (obj, current) if obj is not None else (old, prev)
            permanence_refs = refs(source, selected, 'track_id', 'visibility_state', 'frame_truncation', 'occlusion_state', 'bbox')
            permanence_refs += [_ref(current, field='objects')]
            if prev is not None:
                permanence_refs += [_ref(prev, field='objects')]
            disappeared = old is not None and obj is None
            earlier = [(s, ts[key]) for s, ts in zip(states[:-2], tracks[:-2]) if key in ts] if len(states) >= 3 else []
            reappeared = obj is not None and old is None and bool(earlier)
            if reappeared:
                permanence_refs += refs(earlier[-1][0], earlier[-1][1], 'track_id', 'visibility_state')
            boundary = _attribute(selected, 'frame_truncation')
            occlusion_value = _attribute(selected, 'occlusion_state', possible=True)
            occlusion = None if occlusion_value is None else occlusion_value == 'possible'
            sb = _bbox(selected)
            overlapping = []
            if sb is not None:
                for other in source.objects:
                    bb = _bbox(other)
                    if other.object_id != selected.object_id and bb is not None and min(sb[2], bb[2]) > max(sb[0], bb[0]) and min(sb[3], bb[3]) > max(sb[1], bb[1]):
                        overlapping.append(other.object_id)
                        permanence_refs += refs(source, other, 'bbox', 'visibility_state')
            if disappeared:
                finding = 'possible_frame_exit' if boundary is True else 'possible_occlusion' if occlusion or overlapping else 'unexplained_track_loss'
                status = 'indeterminate'
            elif reappeared:
                finding, status = 'reobserved_track', 'satisfied'
            elif old is not None:
                finding, status = 'object_permanence_consistent', 'satisfied'
            else:
                finding, status = 'no_prior_observation', 'not_applicable'
            measured = {'absent_from_current_observations': disappeared, 'reobserved_after_gap': reappeared,
                        'source_scene_id': source.scene_id, 'frame_truncation': boundary,
                        'occlusion_possible_evidence': occlusion, 'overlapping_observed_object_ids': overlapping}
            emit('object_permanence', (selected.object_id,), status, finding,
                'Observation availability and previous geometry; no state is assigned during a gap.',
                permanence_refs, measured, uncertainty=('track_loss_and_unobserved_state_unresolved',))
            emit('frame_occlusion_consistency', (selected.object_id,),
                'indeterminate' if disappeared or boundary is None else 'satisfied', finding,
                'Truncation/overlap can coexist with observation or later track loss; reobservation does not prove prior occlusion.',
                permanence_refs, measured, uncertainty=('no_confirmed_occlusion_or_exit',))

        bundle = PhysicsConstraintBundle(current.scene_id, current.timestamp, tuple(results))
        provenance = {'session_id': session_id, 'coordinate_frame_id': coordinate_frame_id,
            'history_scene_ids': tuple(s.scene_id for s in states[:-1]), 'policy': policy,
            'timing': 'caller_order_only_no_velocity_for_undated_pairs' if any(s.timestamp is None for s in states) else 'strictly_increasing_source_timestamps',
            'scope': 'sampled_image_plane_observations_only', 'new_physical_objects': False}
        identity = stable_id('transition-assessment', current.scene_id, current.timestamp, plain(bundle), provenance)
        return PhysicsTransitionAssessment(identity, current.scene_id, current.timestamp,
            prev.scene_id if prev else None, prev.timestamp if prev else None, bundle, provenance)
