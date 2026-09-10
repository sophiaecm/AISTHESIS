"""Same-input offline experience ablation. No neural or reasoner calls."""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

from fifth_layer.world_state import WorldState
from fifth_layer.world_model import (WorldStateAdapter, EvidenceItem, EvidenceBundle,
    Hypothesis, HypothesisSet, ExperienceEpisode, ExperienceMemory)
from fifth_layer.world_model.evidence import stable_id
from fifth_layer.world_model.experience_learning import eligibility, experience_context
from .sensory import _plain


def ablate(saved, history, *, top_k=3, max_contribution=.05, capacity=512):
    """Replay only completed past episodes, isolated by run, mode and source.

    Local sequence timestamps from different runs are not a shared chronology.
    Cross-run transfer therefore requires a future explicit time-domain contract
    and is intentionally rejected here. Current frames are never evaluated anew.
    """
    if saved.get('schema_version') != 'evaluation-0.1' or history.get('schema_version') != 'experience-evaluation-0.1':
        raise ValueError('expected saved sensory result and experience report')
    if saved.get('run_id') != history.get('source_run_id'):
        raise ValueError('different run time domains cannot be compared safely')
    groups, past = defaultdict(list), {}
    diagnostics = []
    for session in history['sessions']:
        key = (session['mode'], session['source_sha256'])
        if key in past:
            raise ValueError('duplicate history session')
        episodes = [ExperienceEpisode(**e) for e in session['episodes']]
        if len({e.episode_id for e in episodes}) != len(episodes) or len({e.prediction_id for e in episodes}) != len(episodes):
            raise ValueError('duplicate historical episode/prediction IDs')
        past[key] = episodes
    for record in saved['results']:
        if record['status'] != 'ok' or record['mode'] == 'VLM_ONLY' or not record.get('sensory_world_model'):
            diagnostics.append(dict(mode=record['mode'], sequence=record['input']['sequence'], reason='no_valid_stored_forecast'))
            continue
        groups[(record['mode'], record['input']['source_sha256'])].append(record)
    sessions = []
    for key in sorted(groups):
        records = sorted(groups[key], key=lambda r: (r['input']['timestamp'], r['input']['sequence']))
        if len({r['input']['timestamp'] for r in records}) != len(records):
            raise ValueError('duplicate current frame timestamp')
        replay_time = [0.]
        memory = ExperienceMemory(capacity=capacity, clock=lambda: replay_time[0])
        admitted, frames = set(), []
        for record in records:
            now = record['input']['timestamp']
            eligible = []
            for episode in past.get(key, ()):
                valid, _, known_at = eligibility(episode, current_time=now)
                if valid and episode.episode_id not in admitted:
                    eligible.append((known_at, episode.episode_id, episode))
            for known_at, _, episode in sorted(eligible):
                replay_time[0] = known_at
                memory.add(episode)
                admitted.add(episode.episode_id)
            replay_time[0] = now
            stored = record['sensory_world_model']
            hypotheses = HypothesisSet(**{**stored['hypotheses'], 'hypotheses': tuple(
                Hypothesis(**h) for h in stored['hypotheses']['hypotheses'])})
            evidence = EvidenceBundle(**{**stored['evidence'], 'items': tuple(
                EvidenceItem(**e) for e in stored['evidence']['items'])})
            state = record['fifth_layer_input']
            if state['timestamp'] != now:
                raise ValueError('current input timestamp mismatch')
            scene = WorldStateAdapter.from_world_state(WorldState(now, state['data']), scene_id=hypotheses.scene_id)
            baseline = experience_context(scene, evidence, hypotheses, current_time=now)
            enabled = experience_context(scene, evidence, hypotheses, memory, current_time=now,
                enabled=True, top_k=top_k, max_contribution=max_contribution)
            frames.append(dict(sequence=record['input']['sequence'], timestamp=now,
                current_input_sha256=stable_id('same-input', state, stored),
                NO_EXPERIENCE=baseline, EXPERIENCE_ENABLED=enabled,
                leakage_audit=dict(admitted_before_query=sorted(admitted),
                    admission_policy='source, target, observation and evaluation times strictly before current_time',
                    future_admitted=False),
                affected_hypotheses=[r['hypothesis_id'] for r in enabled['ranking'] if r['experience_contribution'] != 0.]))
        sessions.append(dict(mode=key[0], source_sha256=key[1], frames=frames))
    return _plain(dict(schema_version='experience-learning-ablation-0.1', source_run_id=saved['run_id'],
        sessions=sessions, diagnostics=sorted(diagnostics, key=lambda x: (x['mode'], x['sequence'])),
        policy=dict(top_k=top_k, max_contribution=max_contribution, capacity=capacity),
        interpretation='Heuristic context ablation only; no ground-truth accuracy or improvement claim.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='Saved sensory JSON')
    parser.add_argument('--history', required=True, help='Saved experience JSON from the same run')
    parser.add_argument('--output', required=True, help='New *_experience_learning.json')
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--max-contribution', type=float, default=.05)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() or not output.name.endswith('_experience_learning.json'):
        parser.error('output must be a new *_experience_learning.json file')
    current_bytes, history_bytes = Path(args.input).read_bytes(), Path(args.history).read_bytes()
    result = ablate(json.loads(current_bytes), json.loads(history_bytes),
                    top_k=args.top_k, max_contribution=args.max_contribution)
    result['input_sha256'] = sha256(current_bytes).hexdigest()
    result['history_sha256'] = sha256(history_bytes).hexdigest()
    with output.open('x', encoding='utf-8') as target:
        json.dump(result, target, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
