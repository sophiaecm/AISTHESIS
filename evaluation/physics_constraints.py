"""Offline constraints on saved Physical World Model v0.1 reports only."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from fifth_layer.world_model.physical_state import (
    PhysicalAttribute, PhysicalObjectState, PhysicalRelation, PhysicalWorldState)
from fifth_layer.world_model.physics_constraint_engine import PhysicsConstraintEngine, PhysicsConstraintPolicy


def physical_state_from_dict(data):
    """Strictly rehydrate the frozen v0.1 contract without invoking its builder."""
    if data.get('schema_version') != 'physical-world-0.1':
        raise ValueError('expected physical-world-0.1 snapshot')
    values = dict(data)
    values['objects'] = tuple(PhysicalObjectState(**{**o, 'attributes': {
        k: PhysicalAttribute(**v) for k, v in o['attributes'].items()}}) for o in data['objects'])
    values['relations'] = tuple(PhysicalRelation(**{**r, 'evidence': PhysicalAttribute(**r['evidence'])}) for r in data['relations'])
    return PhysicalWorldState(**values)


def process_saved_result(data, *, policy=None):
    if data.get('schema_version') != 'physical-world-evaluation-0.1':
        raise ValueError('expected physical-world-evaluation-0.1 report')
    engine = PhysicsConstraintEngine(policy)
    sessions = []
    seen = set()
    for session in sorted(data['sessions'], key=lambda s: (s['mode'], s['source_sha256'])):
        key = (session['mode'], session['source_sha256'])
        if key in seen:
            raise ValueError('duplicate saved session')
        seen.add(key)
        states = [physical_state_from_dict(s) for s in session['snapshots']]
        # Preserve serialized order: never repair future/reordered history silently.
        history = []
        assessments = []
        first_seen = {}
        displacement = Counter()
        for state in states:
            for obj in state.objects:
                label = obj.attributes.get('class_name')
                name = label.value if label is not None and label.value is not None else 'unknown'
                first_seen.setdefault(name, state.timestamp)
                d = obj.attributes.get('displacement')
                if d is not None and d.value is not None:
                    displacement[name] += 1
            assessment = engine.assess(state, history=tuple(history), session_id=':'.join(key),
                                       coordinate_frame_id=session['source_sha256'] + ':source-pixels')
            assessments.append(assessment.to_dict())
            history.append(state)
        results = [r for a in assessments for r in a['constraints']['results']]
        violations = [dict(constraint_id=r['constraint_id'], scene_id=r['scene_id'], timestamp=r['timestamp'],
                           object_ids=r['object_ids'], finding=r['finding'], measured_values=r['measured_values'])
                      for r in results if r['status'] == 'violated']
        sessions.append({'mode': key[0], 'source_sha256': key[1], 'assessments': assessments,
            'summary': {'snapshot_count': len(states), 'constraint_count': len(results),
                'source_first_observed_timestamp_by_class': dict(sorted(first_seen.items())),
                'source_displacement_observations_by_class': dict(sorted(displacement.items())),
                'statuses': dict(sorted(Counter(r['status'] for r in results).items())),
                'findings': dict(sorted(Counter(r['finding'] for r in results).items())),
                'violations': violations}})
    return {'schema_version': 'physics-constraints-evaluation-0.2', 'sessions': sessions,
            'scope': 'offline retrospective image-plane constraints; possibilities are not observed physical facts'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input')
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-displacement-pixels', type=float, default=None)
    parser.add_argument('--max-velocity-change-pixels-per-second', type=float, default=None)
    parser.add_argument('--support-gap-pixels', type=float, default=0.)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() or not output.name.endswith('_physics_constraints.json'):
        parser.error('output must be a new *_physics_constraints.json file')
    source = Path(args.input).read_bytes()
    policy = PhysicsConstraintPolicy(args.max_displacement_pixels, args.max_velocity_change_pixels_per_second,
                                     args.support_gap_pixels)
    result = process_saved_result(json.loads(source), policy=policy)
    result['source_file_sha256'] = sha256(source).hexdigest()
    with output.open('x', encoding='utf-8') as target:
        json.dump(result, target, indent=2, sort_keys=True, allow_nan=False)


if __name__ == '__main__':
    main()
