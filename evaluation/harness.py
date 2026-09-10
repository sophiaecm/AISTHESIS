"""Inspectable mode executor. CLI uses a fresh process for each condition."""
from copy import deepcopy
from dataclasses import fields, is_dataclass
from enum import Enum
from hashlib import sha256
import json
import os
from time import perf_counter

from .audit import BASELINE_FIELDS, check_detector, scan, shared_state_audit
from .claims import extract_claims, contribution_claims
from .media import file_hash


class Mode(str, Enum):
    VLM_ONLY = 'VLM_ONLY'
    FIFTH_LAYER_ONLY = 'FIFTH_LAYER_ONLY'
    VLM_PLUS_FIFTH_LAYER = 'VLM_PLUS_FIFTH_LAYER'


def plain(value):
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): plain(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(child) for child in value]
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise ValueError(f'non-JSON structured output: {type(value).__name__}')


def execute(frames, mode, adapters, *, run_id, isolation='in_process_unverified'):
    """Public injected-adapter API is for tests; it cannot guarantee isolation."""
    mode = Mode(mode)
    results = []
    for frame in frames:
        started = perf_counter()
        if file_hash(frame['evaluation_path']) != frame['frame_sha256']:
            raise ValueError('prepared frame hash changed')
        record = dict(schema_version='evaluation-0.1', run_id=run_id, mode=mode.value,
            input={key: value for key, value in frame.items() if key != 'evaluation_path'},
            status='ok', perception=None, vlm_output=None, fifth_layer_output=None,
            fusion_output=None, fifth_layer_input=None, configuration=None, claims={},
            timing=dict(perception_seconds=0., vlm_seconds=0., fifth_layer_seconds=0., fusion_seconds=0.),
            provenance=dict(isolation=isolation, worker_pid=os.getpid(),
                            components={}, input_policy='detector_tracking_geometry_v1'),
            leakage_diagnostics=dict(findings=[], status='not_applicable',
                checks=['detector allowlist', 'reasoner arguments', 'serialized intermediate JSON',
                        'adapter state', 'loaded Fifth Layer module globals', 'media hashes']),
            ground_truth=None, experiment_annotations={})

        def timed(name, call):
            before = perf_counter()
            value = call()
            record['timing'][name] += perf_counter() - before
            return value

        structured = None
        if mode != Mode.VLM_ONLY:
            raw = timed('perception_seconds', lambda: adapters.detector(frame['evaluation_path']))
            record['perception'] = {'raw_detector': plain(raw)}
            findings = check_detector(raw)
            record['leakage_diagnostics']['findings'].extend(findings)
            if findings:
                record['status'] = 'rejected_input'
            else:
                structured = timed('perception_seconds', lambda: adapters.structured(deepcopy(raw), frame['timestamp']))
                record['perception']['structured'] = plain(structured)
            record['provenance']['components']['perception'] = [
                'YoloDetectorPerception', 'TrackMemory', 'extract_motion_evidence',
                'extract_occlusion_evidence', 'PerceptionFusion._build_scene_relations']
        vlm = None
        if mode != Mode.FIFTH_LAYER_ONLY and record['status'] == 'ok':
            vlm = timed('vlm_seconds', lambda: adapters.vlm(frame['evaluation_path']))
            record['vlm_output'] = plain(vlm)
            record['claims']['vlm'] = extract_claims(record['vlm_output'], 'VLM')
            record['provenance']['components']['vlm'] = 'SmolVLMScenePerception.perceive (unchanged)'
        if structured is not None:
            supplied = structured
            if mode == Mode.VLM_PLUS_FIFTH_LAYER:
                supplied = timed('fusion_seconds', lambda: adapters.combine(structured, vlm))
                record['provenance']['components']['fusion'] = 'PerceptionFusion.fuse'
            record['fifth_layer_input'] = plain(supplied)
            if mode == Mode.FIFTH_LAYER_ONLY:
                findings = scan(supplied.data, 'fifth_layer_input')
                findings += [dict(path=key, reason='baseline_field_not_allowlisted')
                             for key in supplied.data if key not in BASELINE_FIELDS]
                # Audit the actual serialized boundary as well as Python objects.
                findings += scan(json.dumps(record['fifth_layer_input']), 'serialized_boundary')
                shared = shared_state_audit(adapters)
                findings += shared['findings']
                record['leakage_diagnostics'].update(shared_state=shared)
                record['leakage_diagnostics']['findings'].extend(findings)
                if findings:
                    record['status'] = 'rejected_input'
            if record['status'] == 'ok':
                record['fifth_layer_output'] = plain(timed('fifth_layer_seconds', lambda: adapters.fifth(deepcopy(supplied))))
                record['claims']['fifth'] = contribution_claims(record['fifth_layer_output'],
                    record['fifth_layer_input'], 'FifthLayer')
                record['provenance']['components']['fifth'] = 'AisthesisOrchestrator.analyze'
                if mode == Mode.FIFTH_LAYER_ONLY:
                    after = shared_state_audit(adapters)
                    record['leakage_diagnostics']['shared_state_after'] = after
                    record['leakage_diagnostics']['findings'].extend(after['findings'])
                    if after['findings']:
                        record['status'] = 'rejected_output'
                if mode == Mode.VLM_PLUS_FIFTH_LAYER:
                    record['fusion_output'] = deepcopy(record['fifth_layer_output'].get('sensor_fusion'))
                    record['claims']['fusion'] = contribution_claims(record['fusion_output'],
                        record['fifth_layer_input'], 'SensorFusionReasoner')
            serialized = json.dumps(record['fifth_layer_input'], sort_keys=True, allow_nan=False)
            record['provenance']['fifth_layer_input_sha256'] = sha256(serialized.encode()).hexdigest()
        if mode == Mode.FIFTH_LAYER_ONLY:
            record['leakage_diagnostics']['status'] = ('rejected' if record['status'] != 'ok'
                else 'no_markers_in_tested_boundary' if isolation == 'fresh_process'
                else 'no_markers_in_tested_boundary_isolation_unverified')
        record['configuration'] = plain(adapters.metadata())
        record['timing']['total_mode_seconds'] = perf_counter() - started
        if file_hash(frame['evaluation_path']) != frame['frame_sha256']:
            raise ValueError('adapter changed shared prepared frame')
        json.dumps(record, allow_nan=False)
        results.append(record)
    return results
