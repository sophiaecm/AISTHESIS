"""Offline experience report from saved sensory frames; never reruns inference."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

from fifth_layer.world_state import WorldState
from fifth_layer.world_model import (
    WorldStateAdapter, EvidenceItem, EvidenceBundle, Hypothesis, HypothesisSet,
    ExperienceMemory, adapt_temporal_trajectories,
)
from fifth_layer.world_model.prediction_observation import (
    prediction_from_hypothesis, outcome_from_scene, evaluate_prediction, experience_episode,
)
from fifth_layer.world_model._structured import number
from .sensory import _plain


def evaluate_sequence(frames, *, capacity=512, time_tolerance_seconds=0., position_tolerance_pixels=0.):
    """Frames contain scene, issued hypotheses, evidence, optional trajectories.

    Caller must supply a single continuous tracking session and coordinate frame.
    Select observations by time only, never by match quality. All hypotheses are
    recorded, including final forecasts with no later observation. Session memory
    uses a replay clock; its bounded/TTL semantics are unchanged. The returned
    audit report can contain more records than the retained session memory.
    """
    number(time_tolerance_seconds, 'time_tolerance_seconds', nonnegative=True)
    number(position_tolerance_pixels, 'position_tolerance_pixels', nonnegative=True)
    frames = list(frames)
    if any(frame['scene'].timestamp is None for frame in frames):
        raise ValueError('sequence frames require timestamps')
    frames.sort(key=lambda frame: (frame['scene'].timestamp, frame['scene'].scene_id))
    if len({f['scene'].timestamp for f in frames}) != len(frames):
        raise ValueError('duplicate frame timestamp in tracking session')
    if len({f['scene'].scene_id for f in frames}) != len(frames):
        raise ValueError('duplicate scene_id in tracking session')
    episodes = []
    seen = set()
    for frame in frames:
        scene, hypotheses, evidence = frame['scene'], frame['hypotheses'], frame['evidence']
        trajectories = frame.get('trajectories')
        if trajectories is None:
            trajectories = adapt_temporal_trajectories(scene, evidence, hypotheses)
        for hypothesis in sorted(hypotheses.hypotheses, key=lambda h: h.hypothesis_id):
            if hypothesis.created_timestamp != scene.timestamp:
                raise ValueError('issued hypothesis time must match its source frame')
            linked = [t for t in trajectories if t.hypothesis_id == hypothesis.hypothesis_id]
            # Preserve competing trajectories as separate records; no winning path.
            for trajectory in sorted(linked, key=lambda t: t.trajectory_id) or [None]:
                prediction = prediction_from_hypothesis(scene, hypothesis, evidence, trajectory)
                if prediction.prediction_id in seen:
                    raise ValueError('duplicate prediction record')
                seen.add(prediction.prediction_id)
                eligible = [f for f in frames if f['scene'].timestamp > prediction.source_timestamp
                            and abs(f['scene'].timestamp - prediction.target_timestamp) <= time_tolerance_seconds + 1e-9]
                target = min(eligible, key=lambda f: (abs(f['scene'].timestamp - prediction.target_timestamp),
                             f['scene'].timestamp, f['scene'].scene_id), default=None)
                outcome = outcome_from_scene(prediction, target['scene'] if target else None,
                    sensory_evidence=target['evidence'] if target else None)
                evaluation = evaluate_prediction(prediction, outcome,
                    time_tolerance_seconds=time_tolerance_seconds,
                    position_tolerance_pixels=position_tolerance_pixels)
                episodes.append(experience_episode(prediction, outcome, evaluation))
    episodes.sort(key=lambda e: (e.evaluated_timestamp if e.evaluated_timestamp is not None
                  else e.prediction_summary['target_timestamp'], e.episode_id))
    replay_time = [0.]
    memory = ExperienceMemory(capacity=capacity, clock=lambda: replay_time[0])
    for episode in episodes:
        replay_time[0] = (episode.evaluated_timestamp if episode.evaluated_timestamp is not None
                          else episode.prediction_summary['target_timestamp'])
        memory.add(episode)
    return dict(episodes=episodes, retained_episode_ids=[e.episode_id for e in memory.recent()],
                memory_summary=memory.summary(), by_status=dict(sorted(Counter(
                    e.evaluation_status.value for e in episodes).items())))


def process_saved_result(result, **policy):
    """Only stored sensory hypotheses are historical forecasts.

    Original results/claims/comparisons are never rewritten. Reports without
    stored world-model forecasts get explicit skipped diagnostics, not invented
    retrospective predictions. Modes/source videos form separate ID scopes.
    """
    if result.get('schema_version') != 'evaluation-0.1':
        raise ValueError('expected evaluation-0.1 input')
    groups = defaultdict(list)
    diagnostics = []
    for record in result['results']:
        identity = dict(mode=record['mode'], sequence=record['input']['sequence'])
        if record['status'] != 'ok' or record['mode'] == 'VLM_ONLY':
            diagnostics.append(dict(**identity, reason='not_a_successful_fifth_layer_frame'))
            continue
        stored = record.get('sensory_world_model')
        if stored is None:
            diagnostics.append(dict(**identity, reason='no_stored_world_model_predictions'))
            continue
        hypotheses = HypothesisSet(**{**stored['hypotheses'], 'hypotheses': tuple(
            Hypothesis(**h) for h in stored['hypotheses']['hypotheses'])})
        evidence = EvidenceBundle(**{**stored['evidence'], 'items': tuple(
            EvidenceItem(**item) for item in stored['evidence']['items'])})
        supplied = record['fifth_layer_input']
        scene = WorldStateAdapter.from_world_state(WorldState(supplied['timestamp'], supplied['data']),
            scene_id=hypotheses.scene_id, snapshot_sequence_id=identity['sequence'],
            provenance={'source': 'saved fifth_layer_input', 'input': record['input']})
        if scene.timestamp != record['input']['timestamp']:
            raise ValueError('source frame and observation timestamps disagree')
        source = record['input']['source_sha256']
        groups[(record['mode'], source)].append(dict(scene=scene, hypotheses=hypotheses, evidence=evidence))
    sessions = [dict(mode=key[0], source_sha256=key[1], **evaluate_sequence(groups[key], **policy))
                for key in sorted(groups)]
    return _plain(dict(schema_version='experience-evaluation-0.1', source_run_id=result['run_id'],
        sessions=sessions, diagnostics=sorted(diagnostics, key=lambda x: (x['mode'], x['sequence'])),
        policy=policy, provenance=dict(learning=False, inference_feedback=False,
            input_interpretation='issued hypotheses from sensory_world_model; observed fifth_layer_input',
            selection='closest later snapshot inside target-time tolerance; earlier tie; no quality selection',
            evaluation_scope='target snapshot state, not event history or calibrated prediction accuracy')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='Saved *_sensory.json input; read only')
    parser.add_argument('--output', required=True, help='New *_experience.json report')
    parser.add_argument('--time-tolerance-seconds', type=float, default=0.)
    parser.add_argument('--position-tolerance-pixels', type=float, default=0.)
    parser.add_argument('--capacity', type=int, default=512)
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists() or not path.name.endswith('_experience.json'):
        parser.error('output must be a new *_experience.json file')
    source = Path(args.input).read_bytes()
    result = process_saved_result(json.loads(source), capacity=args.capacity,
        time_tolerance_seconds=args.time_tolerance_seconds, position_tolerance_pixels=args.position_tolerance_pixels)
    result['source_file_sha256'] = sha256(source).hexdigest()
    with path.open('x', encoding='utf-8') as target:
        json.dump(result, target, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
