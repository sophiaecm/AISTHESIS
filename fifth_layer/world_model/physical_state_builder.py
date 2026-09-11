"""Deterministic, opt-in projection of current structured observations."""
from collections.abc import Mapping
from math import hypot

from ._structured import geometry, number
from .evidence import EvidenceBundle, stable_id
from .evidence_providers import collect_evidence
from .physical_state import PhysicalAttribute as Attribute, PhysicalObjectState, PhysicalRelation, PhysicalWorldState


ATTRIBUTES = ('track_id', 'class_name', 'bbox', 'center', 'size', 'displacement',
              'velocity', 'motion_state', 'motion_direction', 'visibility_state',
              'occlusion_state', 'frame_truncation', 'confidence', 'depth', 'mass',
              'force', 'friction', 'contact', 'support')


def _time_check(value, cutoff):
    """Reject explicitly future observations, including nested source timestamps."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            if (key == 'timestamp' or key.endswith('_timestamp')) and child is not None:
                number(child, key, nonnegative=True)
                if cutoff is None or child > cutoff:
                    raise ValueError('future or unorderable source timestamp')
            elif isinstance(child, (Mapping, tuple, list)):
                _time_check(child, cutoff)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _time_check(child, cutoff)


def _matches(item, record, index):
    if item.track_id is not None:
        return type(item.track_id) is type(record.get('track_id')) and item.track_id == record.get('track_id')
    return item.object_id == record.get('object_id', index)


class PhysicalStateBuilder:
    """History must belong to the same tracking session and pixel coordinate frame.

    No default motion deadband is invented. Existing motion classification is
    retained; history-only classification requires minimum_motion_pixels.
    """
    def __init__(self, *, minimum_motion_pixels=None):
        if minimum_motion_pixels is not None:
            number(minimum_motion_pixels, 'minimum_motion_pixels', nonnegative=True)
        self.minimum_motion_pixels = minimum_motion_pixels

    def build(self, scene, evidence=None, *, previous_scene=None):
        evidence = collect_evidence(scene) if evidence is None else evidence
        if not isinstance(evidence, EvidenceBundle) or evidence.scene_id != scene.scene_id:
            raise ValueError('evidence must belong to current scene')
        _time_check(scene.observed_objects, scene.timestamp)
        for item in evidence.items:
            _time_check({'timestamp': item.timestamp}, scene.timestamp)
            # Forecast payloads are ignored, not mistaken for observations.
            if item.source_type.value in ('motion', 'occlusion', 'sensory'):
                if item.epistemic_status not in ('expected', 'unavailable') and item.evidence_type != 'occlusion_hypotheses':
                    _time_check(item.value, scene.timestamp)
                    _time_check(item.provenance, scene.timestamp)
        if previous_scene is not None:
            if scene.timestamp is not None and previous_scene.timestamp is not None:
                if previous_scene.timestamp >= scene.timestamp:
                    raise ValueError('history must strictly precede current scene')
            elif (previous_scene.snapshot_sequence_id is None or scene.snapshot_sequence_id is None
                  or previous_scene.snapshot_sequence_id >= scene.snapshot_sequence_id):
                raise ValueError('untimed history requires increasing snapshot sequence IDs')
            _time_check(previous_scene.observed_objects, previous_scene.timestamp)
            if (previous_scene.image_width, previous_scene.image_height) != (scene.image_width, scene.image_height):
                raise ValueError('history coordinate dimensions must match')

        objects = []
        records = {}
        used = set()
        for index, record in enumerate(scene.observed_objects):
            # v0.1 accepts synchronized snapshots, not asynchronous sensor fusion.
            if any(record.get(k) is not None and record[k] != scene.timestamp
                   for k in ('timestamp', 'observation_timestamp')):
                raise ValueError('object observation time must match its scene')
            ref = f'{scene.scene_id}.observed_objects[{index}]'
            track = record.get('track_id')
            if track is not None and type(track) not in (int, str):
                raise ValueError('track_id must be integer or string')
            identity = stable_id('physical-object', 'track', track) if track is not None else stable_id('physical-object', scene.scene_id, index)
            attrs = {name: Attribute() for name in ATTRIBUTES}

            def put(name, value, refs=(ref,), rule=None, units=None, status=None):
                if value is not None:
                    attrs[name] = Attribute(value, status or ('estimated' if rule else 'observed'), refs, rule, units)

            for name in ('track_id', 'class_name', 'confidence'):
                put(name, record.get(name))
            put('visibility_state', 'observed')
            box = record.get('box_xyxy', record.get('bbox'))
            box_field = 'box_xyxy' if record.get('box_xyxy') is not None else 'bbox'
            if box is None and record.get('box') is not None:
                x, y, w, h = record['box']
                box = (x, y, x + w, y + h)
                box_field = 'box'
            if box is not None:
                geometry(box, 'bbox', 4)
                put('bbox', box, (ref + '.' + box_field,), rule='xywh to xyxy' if box_field == 'box' else None, units='pixels')
                x1, y1, x2, y2 = box
                put('center', ((x1+x2)/2, (y1+y2)/2), (ref + '.' + box_field,), 'bbox midpoint', 'pixels')
                put('size', (x2-x1, y2-y1), (ref + '.' + box_field,), 'bbox width and height; not 3D geometry', 'pixels')
                if scene.image_width and scene.image_height:
                    put('frame_truncation', x1 <= 0 or y1 <= 0 or x2 >= scene.image_width or y2 >= scene.image_height,
                        (ref + '.' + box_field, scene.scene_id + '.image_width', scene.scene_id + '.image_height'),
                        'boundary touch/crossing; same rule as perception.occlusion')
            elif record.get('center') is not None:
                geometry(record['center'], 'center', 2)
                put('center', record['center'], (ref + '.center',), units='pixels')

            # Only existing motion extractor records, never temporal forecasts.
            motion = [e for e in evidence.items if e.source_type.value == 'motion'
                      and e.evidence_type == 'motion_evidence' and _matches(e, record, index)
                      and e.timestamp == scene.timestamp]
            if len(motion) == 1:
                e = motion[0]
                used.add(e.evidence_id)
                v = e.value
                refs = (e.evidence_id, ref)
                if v.get('dx') is not None and v.get('dy') is not None:
                    put('displacement', (v['dx'], v['dy']), refs, 'stored motion extractor displacement', 'pixels')
                    dt = v.get('delta_time', v.get('dt'))
                    if dt is not None:
                        number(dt, 'motion delta_time')
                        if dt > 0 and scene.timestamp is not None and dt <= scene.timestamp:
                            put('velocity', (v['dx']/dt, v['dy']/dt), refs,
                                'stored displacement / explicit positive elapsed seconds', 'pixels/second')
                state = v.get('motion_state')
                if state in ('stationary', 'moving', 'moving_left', 'moving_right', 'moving_up', 'moving_down'):
                    put('motion_state', 'stationary' if state == 'stationary' else 'moving', refs, 'retain source motion classification/deadband')
                    if state.startswith('moving_'):
                        put('motion_direction', state[7:], refs, 'retain source image-plane direction')
            prior = [] if previous_scene is None or track is None else [r for r in previous_scene.observed_objects
                if type(r.get('track_id')) is type(track) and r.get('track_id') == track]
            if len(prior) > 1:
                raise ValueError('ambiguous historical track association')
            if len(prior) == 1 and prior[0].get('class_name') == record.get('class_name'):
                p = prior[0]
                if any(p.get(k) is not None and p[k] != previous_scene.timestamp
                       for k in ('timestamp', 'observation_timestamp')):
                    raise ValueError('historical observation time must match its scene')
                pb = p.get('box_xyxy', p.get('bbox'))
                if pb is None and p.get('box') is not None:
                    x, y, w, h = p['box']
                    pb = (x, y, x+w, y+h)
                pc = ((pb[0]+pb[2])/2, (pb[1]+pb[3])/2) if pb is not None else p.get('center')
                center = attrs['center'].value
                if pc is not None and center is not None:
                    refs = (ref, f'{previous_scene.scene_id}.observed_objects[track_id={track!r}]')
                    delta = (center[0]-pc[0], center[1]-pc[1])
                    put('displacement', delta, refs, 'current minus previous observed center, same track', 'pixels')
                    if scene.timestamp is not None and previous_scene.timestamp is not None:
                        dt = scene.timestamp - previous_scene.timestamp
                        put('velocity', tuple(d/dt for d in delta), refs + (scene.scene_id+'.timestamp', previous_scene.scene_id+'.timestamp'),
                            'finite difference / positive elapsed seconds; image-plane estimate', 'pixels/second')
            delta = attrs['displacement'].value
            if delta is not None and attrs['motion_state'].value is None and self.minimum_motion_pixels is not None:
                state = 'stationary' if hypot(*delta) <= self.minimum_motion_pixels else 'moving'
                put('motion_state', state, attrs['displacement'].derived_from,
                    f'displacement magnitude <= configured {self.minimum_motion_pixels} pixels')
            if delta is not None and hypot(*delta) > 0 and attrs['motion_direction'].value is None:
                put('motion_direction', tuple(d/hypot(*delta) for d in delta), attrs['displacement'].derived_from,
                    'unit image-plane displacement vector; may include sub-deadband jitter')
            for e in evidence.items:
                if (e.source_type.value == 'occlusion' and e.evidence_type == 'occlusion_evidence'
                        and e.timestamp == scene.timestamp and _matches(e, record, index)):
                    if e.value.get('has_overlap_evidence') is True:
                        put('occlusion_state', 'possible', (ref, e.evidence_id), 'stored overlap geometry only', status='possible')
                        used.add(e.evidence_id)
                    if attrs['frame_truncation'].value is None and type(e.value.get('frame_truncated')) is bool:
                        put('frame_truncation', e.value['frame_truncated'], (ref, e.evidence_id), 'stored boundary geometry')
                        used.add(e.evidence_id)
            objects.append(PhysicalObjectState(identity, scene.timestamp, attrs, {'observation_ref': ref, 'raw_observation': record}))
            records[index] = identity

        relations = []
        for a in objects:
            for b in objects:
                if a.object_id == b.object_id:
                    continue
                ac, bc = a.attributes['center'].value, b.attributes['center'].value
                refs = a.attributes['center'].derived_from + b.attributes['center'].derived_from
                if ac is not None and bc is not None:
                    for valid, kind in ((ac[0] < bc[0], 'left_of'), (ac[0] > bc[0], 'right_of'),
                                        (ac[1] < bc[1], 'above'), (ac[1] > bc[1], 'below')):
                        if valid:
                            relations.append(PhysicalRelation(a.object_id, kind, b.object_id,
                                Attribute(True, 'estimated', refs, 'compare image-plane centers; y increases down')))
                ab, bb = a.attributes['bbox'].value, b.attributes['bbox'].value
                if ab is not None and bb is not None and min(ab[2], bb[2]) > max(ab[0], bb[0]) and min(ab[3], bb[3]) > max(ab[1], bb[1]):
                    relations.append(PhysicalRelation(a.object_id, 'overlaps', b.object_id,
                        Attribute(True, 'estimated', refs, 'positive bbox intersection area; no causal implication')))
        # Explicit observed relation contract. No free text or expected sensations.
        for e in evidence.items:
            if (e.source_type.value == 'sensory' and e.epistemic_status == 'observed'
                    and e.evidence_type in ('contact_observation', 'support_observation') and e.timestamp == scene.timestamp):
                v = e.value
                first, second = records.get(v.get('subject_object_id')), records.get(v.get('target_object_id'))
                if first is not None and second is not None and first != second and v.get('observed') is True:
                    kind = 'contact_possible' if e.evidence_type == 'contact_observation' else 'support_possible'
                    relations.append(PhysicalRelation(first, kind, second,
                        Attribute(True, 'possible', (e.evidence_id,), 'explicit observed structured relation; no dynamics claim')))
                    used.add(e.evidence_id)
        return PhysicalWorldState(scene.scene_id, scene.timestamp, tuple(objects), tuple(relations),
            tuple(e.evidence_id for e in evidence.items),
            {'missing_objects': 'not observed in this snapshot; disappearance is unknown',
             'geometry': '2D image plane only', 'timing': 'unknown' if scene.timestamp is None else 'source timestamp'},
            {'builder': 'PhysicalStateBuilder v0.1', 'minimum_motion_pixels': self.minimum_motion_pixels,
             'used_evidence_ids': sorted(used), 'context_only_evidence_ids': sorted(e.evidence_id for e in evidence.items if e.evidence_id not in used),
             'used_evidence': [e for e in evidence.items if e.evidence_id in used],
             'scene_provenance': scene.provenance})
