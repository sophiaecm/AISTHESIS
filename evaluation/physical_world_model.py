"""Transform stored Case A/B observations only; no inference or model calls."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

from fifth_layer.world_model import SceneState
from fifth_layer.world_model.evidence import stable_id
from fifth_layer.world_model.physical_state_builder import PhysicalStateBuilder


def process_saved_result(result):
    if result.get('schema_version') != 'evaluation-0.1':
        raise ValueError('expected evaluation-0.1 input')
    groups = defaultdict(list)
    diagnostics = []
    for record in result['results']:
        meta = record['input']
        if record.get('status') != 'ok' or not record.get('fifth_layer_input'):
            diagnostics.append({'mode': record['mode'], 'sequence': meta['sequence'], 'reason': 'no successful stored observation'})
            continue
        supplied = record['fifth_layer_input']
        if supplied['timestamp'] != meta['timestamp']:
            raise ValueError('stored observation and frame timestamps disagree')
        data = supplied['data']
        # Deliberately project only observed geometry, not reasoner/VLM outputs.
        observations = data.get('accepted_detections', data.get('detections', ())) or ()
        observations = tuple(o for o in observations if not o.get('is_predicted') and o.get('observation_state', 'observed') == 'observed')
        scene = SceneState(stable_id('physical-scene', record['mode'], meta['source_sha256'], meta['sequence']),
            timestamp=meta['timestamp'], snapshot_sequence_id=meta['sequence'],
            image_width=data.get('image_width'), image_height=data.get('image_height'),
            observed_objects=observations, motion_evidence=data.get('motion_evidence', ()) or (),
            occlusion_evidence={'occlusion_evidence': data.get('occlusion_evidence', ()) or ()},
            provenance={'source': 'stored fifth_layer_input', 'input': meta, 'mode': record['mode']})
        groups[(record['mode'], meta['source_sha256'])].append(scene)
    sessions = []
    for key, scenes in sorted(groups.items()):
        scenes.sort(key=lambda s: (s.timestamp, s.snapshot_sequence_id))
        previous = None
        snapshots = []
        for scene in scenes:
            state = PhysicalStateBuilder().build(scene, previous_scene=previous)
            snapshots.append(state.to_dict())
            previous = scene
        counts = Counter(o['attributes']['class_name']['value'] or 'unknown' for s in snapshots for o in s['objects'])
        first_seen = {}
        motion_counts = Counter()
        for s in snapshots:
            for o in s['objects']:
                attrs = o['attributes']
                name = attrs['class_name']['value'] or 'unknown'
                first_seen.setdefault(name, s['timestamp'])
                if attrs['displacement']['value'] is not None:
                    motion_counts[name] += 1
        sessions.append({'mode': key[0], 'source_sha256': key[1], 'snapshots': snapshots,
            'summary': {'snapshot_count': len(snapshots), 'object_observations_by_class': dict(sorted(counts.items())),
                        'first_observed_timestamp_by_class': dict(sorted(first_seen.items())),
                        'displacement_observations_by_class': dict(sorted(motion_counts.items())),
                        'relation_types': sorted({r['relation'] for s in snapshots for r in s['relations']})}})
    return {'schema_version': 'physical-world-evaluation-0.1', 'sessions': sessions,
            'diagnostics': sorted(diagnostics, key=lambda d: (d['mode'], d['sequence'])),
            'scope': 'architectural validation only; no accuracy claim; stored observations only'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() or not output.name.endswith('_physical_world.json'):
        parser.error('output must be a new *_physical_world.json file')
    source = Path(args.input).read_bytes()
    report = process_saved_result(json.loads(source))
    report['source_file_sha256'] = sha256(source).hexdigest()
    with output.open('x', encoding='utf-8') as target:
        json.dump(report, target, indent=2, sort_keys=True, allow_nan=False)


if __name__ == '__main__':
    main()
