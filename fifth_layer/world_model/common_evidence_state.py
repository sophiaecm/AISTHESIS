"""Bounded, immutable evidence inventory. Evidence aggregation is not belief formation."""
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
import json

from ._structured import freeze, identifier, number
from .evidence import EvidenceBundle, EvidenceItem


FAMILIES = ('visual', 'physical', 'physics_constraint', 'learned_representation',
            'sensory', 'documentary', 'physics', 'temporal', 'occlusion',
            'semantic', 'motion', 'tracking', 'experience')
FRAME_BOUND = ('visual', 'physical', 'physics_constraint', 'learned_representation',
               'physics', 'motion', 'tracking', 'occlusion')


def bounded_plain(value):
    """Canonical JSON data with a bounded traversal, rejecting opaque payloads."""
    remaining = 50000
    def visit(v, depth=0):
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > 24:
            raise ValueError('common metadata exceeds node/depth limit')
        if isinstance(v, Enum):
            v = v.value
        if v is None or type(v) in (str, int, float, bool):
            if isinstance(v, str) and len(v) > 16384:
                raise ValueError('common metadata string too large')
            if type(v) in (int, float):
                number(v, 'metadata')
            return v
        if is_dataclass(v) and not isinstance(v, type):
            v = {f.name: getattr(v, f.name) for f in fields(v)}
        if isinstance(v, Mapping):
            result = {}
            for key, child in v.items():
                if type(key) is not str or len(key) > 256:
                    raise ValueError('invalid metadata key')
                if key.lower() in ('embedding', 'embeddings', 'vector', 'tensor', 'array', 'model', 'file_handle'):
                    raise ValueError('raw payload field forbidden; supply a reference')
                result[key] = visit(child, depth + 1)
            return dict(sorted(result.items()))
        if isinstance(v, (list, tuple, set, frozenset)):
            result = [visit(child, depth + 1) for child in v]
            return sorted(result, key=lambda x: json.dumps(x, sort_keys=True)) if isinstance(v, (set, frozenset)) else result
        raise ValueError('opaque payload forbidden')
    result = visit(value)
    if len(json.dumps(result, sort_keys=True, allow_nan=False)) > 1048576:
        raise ValueError('common metadata exceeds 1 MiB serialized limit')
    return result


def check_context(data, *, scene_id, session_id, timestamp, coordinate_frame_id,
                  frame_bound=False, source_time=None):
    """Declared source times, not prediction target times, must precede cutoff."""
    issues = set()
    if source_time is None:
        issues.add('temporal_alignment_unknown')
    elif timestamp is None or source_time > timestamp:
        raise ValueError('future or unorderable evidence timestamp')
    def visit(v):
        if isinstance(v, Mapping):
            for key, child in v.items():
                if key in ('session_id', 'scene_id') and child is not None:
                    if child != {'session_id': session_id, 'scene_id': scene_id}[key]:
                        raise ValueError(f'{key} mismatch')
                if key == 'coordinate_frame_id' and child is not None and frame_bound:
                    if coordinate_frame_id is None:
                        issues.add('coordinate_frame_unspecified')
                    elif child != coordinate_frame_id:
                        raise ValueError('coordinate frame mismatch')
                if (key == 'timestamp' or key.endswith('_timestamp')) and key != 'target_timestamp' and child is not None:
                    number(child, key, nonnegative=True)
                    cutoff = timestamp if source_time is None else source_time
                    if cutoff is None or child > cutoff:
                        raise ValueError('future or impossible source timestamp ordering')
                visit(child)
        elif isinstance(v, (tuple, list)):
            for child in v:
                visit(child)
    visit(data)
    return issues


@dataclass(frozen=True)
class CommonEvidenceState:
    scene_id: str
    session_id: str
    timestamp: float | None
    evidence: EvidenceBundle
    coordinate_frame_id: str | None = None
    present_families: tuple[str, ...] = ()
    provenance: Mapping = field(default_factory=dict)
    source_references: Mapping = field(init=False)
    availability: Mapping = field(init=False)
    uncertainty: Mapping = field(init=False)
    integrity: Mapping = field(init=False)
    schema_version: str = field(default='common-evidence-state-0.1', init=False)

    def __post_init__(self):
        for name in ('scene_id', 'session_id', 'coordinate_frame_id'):
            identifier(getattr(self, name), name, optional=name == 'coordinate_frame_id')
        if self.timestamp is not None:
            number(self.timestamp, 'timestamp', nonnegative=True)
        if not isinstance(self.evidence, EvidenceBundle) or self.evidence.scene_id != self.scene_id:
            raise ValueError('common state requires a matching EvidenceBundle')
        if not isinstance(self.present_families, (tuple, list)) or any(f not in FAMILIES for f in self.present_families):
            raise ValueError('unsupported present family')
        # Reconstruct a detached bundle without altering any existing evidence fields.
        data = bounded_plain(self.evidence)
        items = tuple(EvidenceItem(**item) for item in data['items'])
        bundle = EvidenceBundle(self.scene_id, items, data['provenance'])
        object.__setattr__(self, 'evidence', bundle)
        context = dict(scene_id=self.scene_id, session_id=self.session_id,
                       timestamp=self.timestamp, coordinate_frame_id=self.coordinate_frame_id)
        integration = check_context(data['provenance'], **context, source_time=self.timestamp,
                                    frame_bound=any(i.source_type.value in FRAME_BOUND for i in items))
        provenance = bounded_plain(self.provenance)
        if not isinstance(provenance, Mapping):
            raise ValueError('common provenance must be a mapping')
        integration.update(check_context(provenance, **context, source_time=self.timestamp))
        references, source_uncertainty = {}, {}
        availability = {family: 'unavailable' for family in FAMILIES}
        for family in self.present_families:
            availability[family] = 'available'
        for item in items:
            family = item.source_type.value
            if family == 'learned_representation' and (self.timestamp is None or item.timestamp != self.timestamp):
                raise ValueError('learned timestamp must match known integration timestamp')
            item_data = bounded_plain(item)
            integration.update(check_context(item_data, **context, source_time=item.timestamp,
                                             frame_bound=family in FRAME_BOUND))
            if item.provenance.get('session_id') is None:
                integration.add('source_context_incomplete')
            if family in FRAME_BOUND and (self.coordinate_frame_id is None or item.provenance.get('coordinate_frame_id') is None):
                integration.add('coordinate_frame_unspecified')
            if item.epistemic_status != 'unavailable':
                availability[family] = 'available'
            references[item.evidence_id] = {
                'source_family': family, 'source_type': item.evidence_type,
                'source_id': item.evidence_id, 'producer': item.source_component,
                'epistemic_status': item.epistemic_status,
                'availability': 'unavailable' if item.epistemic_status == 'unavailable' else 'available',
                'scene_id': item.scene_id, 'timestamp': item.timestamp,
                'session_id': item.provenance.get('session_id'),
                'provenance': item.provenance}
            # Preserve uncertainty at its original path, without pooling or interpreting it.
            found = {}
            def collect(v, path):
                if isinstance(v, Mapping):
                    for key, child in v.items():
                        if 'uncertainty' in key:
                            found[path + '.' + key] = child
                        collect(child, path + '.' + key)
                elif isinstance(v, (list, tuple)):
                    for index, child in enumerate(v):
                        collect(child, f'{path}[{index}]')
            collect(item_data, 'item')
            source_uncertainty[item.evidence_id] = found
        object.__setattr__(self, 'present_families', tuple(sorted(set(self.present_families))))
        for name, value in (
            ('provenance', provenance), ('source_references', references), ('availability', availability),
            ('uncertainty', {'sources': source_uncertainty, 'integration': tuple(sorted(integration))}),
            ('integrity', {'duplicate_policy': 'reject_all_duplicate_ids',
                           'context_policy': 'reject_declared_mismatch_flag_missing_context',
                           'truth_decision': 'not_performed'})):
            object.__setattr__(self, name, freeze(value))
        bounded_plain(self)

    @property
    def evidence_items(self):
        return self.evidence.items

    def to_dict(self):
        return bounded_plain(self)

    def to_json(self):
        return json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'), allow_nan=False)
