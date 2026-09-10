"""Fail-closed baseline boundary and deliberately bounded leakage diagnostics."""
import json
import sys
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass


FORBIDDEN = ('scene_description', 'semantic_evidence', 'semantic_text', 'smolvlm',
             'vlm_output', 'vlm_cache', 'latent_hypothesis', 'predicted_event',
             'state_json', 'motion_json', 'snapshot', 'provenance')
DETECTOR_FIELDS = frozenset({'source_type', 'source_path', 'image_width', 'image_height',
                           'detector', 'model', 'detections', 'detection_count'})
DETECTION_FIELDS = frozenset({'class_id', 'class_name', 'confidence', 'box_xyxy', 'box', 'object_id'})
BASELINE_FIELDS = frozenset({'image_width', 'image_height', 'detections', 'accepted_detections',
                            'detection_count', 'predicted_tracks', 'motion_evidence',
                            'scene_relations', 'occlusion_evidence'})


def scan(value, path='arguments', depth=0):
    """Inspect dictionaries, dataclasses, sequences and serialized JSON strings.

    This detects field/provenance markers, not semantic origins of arbitrary labels.
    Opaque neural/native objects are not introspected and are reported separately.
    """
    if depth > 24:
        return [dict(path=path, reason='audit_depth_limit')]
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    findings = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            location = f'{path}.{key}'
            if any(marker in str(key).lower() for marker in FORBIDDEN):
                findings.append(dict(path=location, reason='VLM_or_opaque_state_field'))
            findings.extend(scan(child, location, depth + 1))
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            findings.extend(scan(child, f'{path}[{index}]', depth + 1))
    elif isinstance(value, str):
        lower = value.lower()
        if 'smolvlm' in lower or 'vlm_generated' in lower:
            findings.append(dict(path=path, reason='VLM_origin_marker'))
        if value.lstrip().startswith(('{', '[')):
            try:
                decoded = json.loads(value)
            except (ValueError, TypeError):
                pass
            else:
                findings.extend(scan(decoded, path + '.serialized', depth + 1))
    return findings


def check_detector(state):
    findings = scan(state.data)
    for key in state.data:
        if key not in DETECTOR_FIELDS:
            findings.append(dict(path=f'detector.{key}', reason='field_not_allowlisted'))
    for index, detection in enumerate(state.data.get('detections', [])):
        for key in detection:
            if key not in DETECTION_FIELDS:
                findings.append(dict(path=f'detections[{index}].{key}', reason='field_not_allowlisted'))
    return findings


def shared_state_audit(adapter):
    # Configuration may legitimately name both model paths; it is not evidence.
    state = {key: value for key, value in getattr(adapter, '__dict__', {}).items() if key != 'config'}
    findings = scan(state, 'adapter_state')
    modules = []
    for name, module in sorted(tuple(sys.modules.items())):
        if not name.startswith('fifth_layer') or module is None:
            continue
        modules.append(name)
        if 'smolvlm' in name:
            findings.append(dict(path=name, reason='VLM_module_loaded_in_baseline_process'))
        for key, value in vars(module).copy().items():
            if key.startswith('__') or not isinstance(value, (dict, list, tuple)):
                continue
            findings.extend(scan({key: value}, 'module_globals.' + name))
    return dict(findings=findings, inspected_modules=modules,
                limitations=['Opaque model/native state and arbitrary label origins cannot be proven clean.',
                             'Fresh processes isolate Python state, not external services or OS/GPU caches.',
                             'Marker checks cannot identify all paraphrased or deliberately disguised VLM content.'])
