"""Join saved physical and constraint reports without rerunning either engine."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from evaluation.physics_constraints import physical_state_from_dict
from fifth_layer.world_model.physics_constraints import PhysicsConstraintResult, PhysicsConstraintBundle, PhysicsTransitionAssessment
from fifth_layer.world_model.latent_physical_state_builder import LatentPhysicalStateBuilder


def assessment_from_dict(data):
    bundle = data['constraints']
    return PhysicsTransitionAssessment(**{**data, 'constraints': PhysicsConstraintBundle(
        **{**bundle, 'results': tuple(PhysicsConstraintResult(**r) for r in bundle['results'])})})


def process_saved_results(physical_report, constraint_report, *, physical_source_sha256, constraint_source_sha256):
    for digest in (physical_source_sha256, constraint_source_sha256):
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('source SHA-256 must be lowercase hexadecimal')
    if physical_report.get('schema_version') != 'physical-world-evaluation-0.1':
        raise ValueError('expected physical-world-evaluation-0.1')
    if constraint_report.get('schema_version') != 'physics-constraints-evaluation-0.2':
        raise ValueError('expected physics-constraints-evaluation-0.2')
    if constraint_report.get('source_file_sha256') != physical_source_sha256:
        raise ValueError('constraint source SHA does not match physical input bytes')

    def sessions(report):
        result = {}
        for session in report['sessions']:
            key = (session['mode'], session['source_sha256'])
            if key in result:
                raise ValueError('duplicate session')
            result[key] = session
        return result

    physical_sessions, constraint_sessions = sessions(physical_report), sessions(constraint_report)
    if set(physical_sessions) != set(constraint_sessions):
        raise ValueError('physical and constraint session sets must match')
    output = []
    for key, session in sorted(physical_sessions.items()):
        source_states = session['snapshots']
        assessments = constraint_sessions[key]['assessments']
        if len(source_states) != len(assessments):
            raise ValueError('one aligned assessment required per physical snapshot')
        history = []
        first_observed = {}
        displacement = Counter()
        signals = Counter()
        for p, a in zip(source_states, assessments):
            physical = physical_state_from_dict(p)
            assessment = assessment_from_dict(a)
            latent = LatentPhysicalStateBuilder().build(physical, assessment, history=history,
                session_id=':'.join(key), coordinate_frame_id=key[1] + ':source-pixels')
            history.append(latent)
            for obj in latent.objects:
                attrs = obj['attributes']
                name = attrs['class_name']['value'] or 'unknown'
                first_observed.setdefault(name, latent.timestamp)
                if attrs['displacement']['value'] is not None:
                    displacement[name] += 1
            signals.update(r['signal'] for r in latent.relations)
        output.append({'mode': key[0], 'source_sha256': key[1],
            'snapshots': [s.to_dict() for s in history],
            'summary': {'snapshot_count': len(history), 'first_observed_timestamp_by_class': dict(sorted(first_observed.items())),
                'displacement_observations_by_class': dict(sorted(displacement.items())), 'relation_signals': dict(sorted(signals.items())),
                'uncertainty_explicit_in_all_snapshots': all(s.uncertainty for s in history)}})
    return {'schema_version': 'latent-physical-evaluation-0.1', 'sessions': output,
        'source_files': {'physical_world_sha256': physical_source_sha256, 'physics_constraints_sha256': constraint_source_sha256},
        'scope': 'offline structured compression only; no learned embeddings or new physical facts'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('physical_input')
    parser.add_argument('constraint_input')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    target = Path(args.output)
    if target.exists() or not target.name.endswith('_latent_physical_state.json'):
        parser.error('output must be a new *_latent_physical_state.json file')
    physical = Path(args.physical_input).read_bytes()
    constraints = Path(args.constraint_input).read_bytes()
    result = process_saved_results(json.loads(physical), json.loads(constraints),
        physical_source_sha256=sha256(physical).hexdigest(), constraint_source_sha256=sha256(constraints).hexdigest())
    with target.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)


if __name__ == '__main__':
    main()
