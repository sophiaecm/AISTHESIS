"""Opt-in postprocessing of harness output; no media, neural or baseline writes."""
import argparse
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import fields, is_dataclass
from enum import Enum
import json
from pathlib import Path

from fifth_layer.world_state import WorldState
from fifth_layer.reasoners.auditory import AuditoryReasoner
from fifth_layer.world_model import WorldStateAdapter, collect_evidence, MultiHypothesisGenerator


def _plain(value):
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {key: _plain(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(child) for child in value]
    return value


def enrich(result):
    """Retain original claims/comparisons; add separately labelled structured data.

    Calls only the existing deterministic auditory reasoner, outside the evidence
    provider. No synthetic physics inputs, track bindings or new sensory chains.
    """
    if result.get('schema_version') != 'evaluation-0.1':
        raise ValueError('expected evaluation-0.1 harness output')
    if 'sensory_integration' in result:
        raise ValueError('result already contains sensory integration')
    enriched = deepcopy(result)
    for record in enriched['results']:
        if record['mode'] == 'VLM_ONLY' or record['status'] != 'ok':
            continue
        supplied = record.get('fifth_layer_input')
        if not isinstance(supplied, dict) or 'data' not in supplied or 'timestamp' not in supplied:
            raise ValueError('successful Fifth Layer record requires original WorldState input')
        state = WorldState(timestamp=supplied['timestamp'], data=supplied['data'])
        sequence = record['input']['sequence']
        scene = WorldStateAdapter.from_world_state(state,
            scene_id=f"{result['run_id']}:{record['mode']}:{sequence}",
            provenance={'source': 'evaluation.fifth_layer_input', 'sequence': sequence})
        output = AuditoryReasoner().infer_expected_consequences(state)
        branches = record.get('fifth_layer_output') or {}
        evidence = collect_evidence(scene, auditory=output, include_sensory=True,
            temporal=branches.get('temporal', {}).get('expected'),
            occlusion=branches.get('occlusion', {}).get('expected'))
        hypotheses = MultiHypothesisGenerator().generate(scene, evidence)
        record['sensory_world_model'] = _plain(dict(
            auditory_expected=output.predictions, evidence=evidence, hypotheses=hypotheses))
    enriched['sensory_integration'] = dict(version='sensory-0.1',
        source='postprocessing original per-frame inputs and reasoner outputs',
        observations_supplied=False, missing_reasoners=['tactile', 'thermal', 'kinesthetic'],
        comparison_policy='original claims and comparisons unchanged; sensory data is separate',
        interpretation='Structured coverage only; no calibrated probability or measured benefit.')
    return enriched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='Previously saved harness output, read only')
    parser.add_argument('--output', required=True, help='New *_sensory.json file')
    args = parser.parse_args()
    path = Path(args.output)
    if not path.name.endswith('_sensory.json') or path.exists():
        parser.error('output must be a new *_sensory.json file')
    with open(args.input, encoding='utf-8') as source:
        result = enrich(json.load(source))
    with path.open('x', encoding='utf-8') as target:
        json.dump(result, target, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
