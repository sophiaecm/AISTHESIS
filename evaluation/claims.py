"""Transparent exact normalized atoms, not semantic equivalence or quality."""
import re


METADATA = frozenset({'timestamp', 'source_path', 'source_type', 'model', 'device',
    'perception_module', 'perception_sources', 'prediction_id', 'track_id', 'object_id',
    'provenance', 'source', 'reasoner', 'detector'})


def extract_claims(value, component, path=''):
    result = []
    if isinstance(value, dict):
        for key in sorted(value):
            if key in METADATA or key.endswith('_timestamp'):
                continue
            result.extend(extract_claims(value[key], component, f'{path}.{key}' if path else key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.extend(extract_claims(child, component, f'{path}[{index}]'))
    elif isinstance(value, str) and value.strip():
        result.append(dict(atom=re.sub(r'\s+', ' ', value).strip().casefold(),
                           component=component, source_path=path, kind='exact_normalized_text'))
    elif type(value) is bool:
        result.append(dict(atom=f'{path.rsplit(".", 1)[-1]}={str(value).lower()}',
                           component=component, source_path=path, kind='structured_boolean'))
    return result


def contribution_claims(output, inputs, component):
    inherited = {claim['atom'] for claim in extract_claims(inputs, 'input')}
    claims = extract_claims(output, component)
    for claim in claims:
        claim['also_in_input'] = claim['atom'] in inherited
    return claims


def compare(records):
    """Compare one frame across modes, rejecting different media identities."""
    if not records:
        raise ValueError('comparison requires records')
    if any(record['input'] != records[0]['input'] for record in records):
        raise ValueError('comparison requires the same source and selected frame')
    if len({r['mode'] for r in records}) != len(records):
        raise ValueError('duplicate comparison mode')
    modes = {r['mode']: r for r in records if r['status'] == 'ok'}
    def atoms(mode, field):
        return {c['atom'] for c in modes.get(mode, {}).get('claims', {}).get(field, [])
                if not c.get('also_in_input', False)}
    vlm = atoms('VLM_ONLY', 'vlm')
    fifth = atoms('FIFTH_LAYER_ONLY', 'fifth')
    combined = atoms('VLM_PLUS_FIFTH_LAYER', 'fifth') | atoms('VLM_PLUS_FIFTH_LAYER', 'fusion')
    required = {'VLM_ONLY', 'FIFTH_LAYER_ONLY', 'VLM_PLUS_FIFTH_LAYER'}
    return dict(input=records[0]['input'], comparable=required <= modes.keys(),
        available_modes=sorted(modes), shared=sorted(vlm & fifth),
        vlm_only=sorted(vlm - fifth), fifth_layer_only=sorted(fifth - vlm),
        combined_only=sorted(combined - vlm - fifth),
        interpretation='Exact normalized output differences only; no semantic equivalence, causal attribution or quality claim.',
        limitation='Copied input atoms are excluded from inferred contributions; this conservative filter may omit genuinely recomputed facts.')
