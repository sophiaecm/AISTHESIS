"""Immutable, source-labelled evidence for opt-in world-model reasoning."""
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
import json
from collections.abc import Mapping

from ._structured import freeze, freeze_fields, identifier, number, timestamps


def stable_id(prefix, *parts):
    """Canonical structured content identity; no wall clock or random UUID."""
    def plain(value):
        if isinstance(value, Mapping):
            return {key: plain(child) for key, child in value.items()}
        if isinstance(value, (tuple, list)):
            return [plain(child) for child in value]
        if isinstance(value, (set, frozenset)):
            return sorted((plain(child) for child in value), key=lambda x: json.dumps(x, sort_keys=True))
        return value
    payload = json.dumps(plain(parts), sort_keys=True, separators=(',', ':'), allow_nan=False)
    return prefix + ':' + sha256(payload.encode('utf-8')).hexdigest()


class EvidenceSource(str, Enum):
    PHYSICS = 'physics'
    TEMPORAL = 'temporal'
    OCCLUSION = 'occlusion'
    SEMANTIC = 'semantic'
    MOTION = 'motion'
    TRACKING = 'tracking'
    SENSORY = 'sensory'
    EXPERIENCE = 'experience'
    VISUAL = 'visual'
    PHYSICAL = 'physical'
    PHYSICS_CONSTRAINT = 'physics_constraint'
    LEARNED_REPRESENTATION = 'learned_representation'
    DOCUMENTARY = 'documentary'


# Opt-in transport sources; existing provider validation is unchanged.
INTEGRATION_STATUSES = {
    EvidenceSource.VISUAL: ('observed', 'estimated', 'possible', 'unknown', 'unavailable'),
    EvidenceSource.PHYSICAL: ('mixed', 'observed', 'estimated', 'possible', 'unknown', 'unavailable'),
    EvidenceSource.PHYSICS_CONSTRAINT: ('assessment',),
    EvidenceSource.LEARNED_REPRESENTATION: ('learned_signal',),
    EvidenceSource.DOCUMENTARY: ('reported_result', 'author_claim', 'aisthesis_inference',
        'unknown', 'limitation', 'method', 'measurement', 'dataset_reference',
        'figure_evidence', 'table_evidence', 'equation_relation'),
}


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    scene_id: str
    source_type: EvidenceSource
    source_component: str
    evidence_type: str
    value: Mapping
    timestamp: float | None
    confidence: float | None = None
    supports: tuple[str, ...] = ()
    contradicts: tuple[str, ...] = ()
    track_id: int | str | None = None
    object_id: int | str | None = None
    provenance: Mapping = field(default_factory=dict)
    modality: str | None = None
    epistemic_status: str | None = None
    supporting_evidence_ids: tuple[str, ...] = ()
    opposing_evidence_ids: tuple[str, ...] = ()

    def __post_init__(self):
        for name in ('evidence_id', 'scene_id', 'source_component', 'evidence_type'):
            identifier(getattr(self, name), name)
        try:
            object.__setattr__(self, 'source_type', EvidenceSource(self.source_type))
        except (ValueError, TypeError) as exc:
            raise ValueError('unsupported evidence source_type') from exc
        if not isinstance(self.value, Mapping):
            raise ValueError('value must be a structured mapping')
        if self.source_type == EvidenceSource.SENSORY:
            if self.modality not in ('auditory', 'tactile', 'thermal', 'kinesthetic'):
                raise ValueError('sensory evidence requires an explicit supported modality')
            if self.epistemic_status not in ('observed', 'inferred', 'expected', 'unavailable'):
                raise ValueError('sensory evidence requires an explicit epistemic_status')
            if self.epistemic_status == 'unavailable' and (
                    self.supports or self.contradicts or self.confidence is not None
                    or self.supporting_evidence_ids or self.opposing_evidence_ids):
                raise ValueError('unavailable evidence cannot support or oppose claims')
            if self.epistemic_status == 'unavailable' and dict(self.value) != {'observation_available': False}:
                raise ValueError('unavailable evidence describes availability only')
            observed = self.value.get('observed')
            if observed is not None and (type(observed) is not bool or
                    observed != (self.epistemic_status == 'observed')):
                raise ValueError('observed flag conflicts with epistemic_status')
        elif self.source_type in INTEGRATION_STATUSES:
            if self.modality is not None or self.epistemic_status not in INTEGRATION_STATUSES[self.source_type]:
                raise ValueError('integration source requires a compatible explicit epistemic_status')
        elif self.modality is not None or self.epistemic_status is not None:
            raise ValueError('modality and epistemic_status require sensory source_type')
        for name in ('track_id', 'object_id'):
            value = getattr(self, name)
            if value is not None and type(value) not in (str, int):
                raise ValueError(f'{name} must be an integer, string or None')
        timestamps(self, ('timestamp',), optional=('timestamp',))
        if self.confidence is not None:
            number(self.confidence, 'confidence', unit=True)
        names = ('value', 'supports', 'contradicts', 'provenance',
                 'supporting_evidence_ids', 'opposing_evidence_ids')
        if self.source_type in INTEGRATION_STATUSES:
            # Structured physical uncertainty is not a scalar probability.
            for name in names:
                object.__setattr__(self, name, freeze(getattr(self, name), name))
            if not isinstance(self.provenance, Mapping):
                raise ValueError('provenance must be a structured mapping')
        else:
            freeze_fields(self, names)
        if not isinstance(self.value, Mapping):
            raise ValueError('value must be a structured mapping')
        for name in ('supports', 'contradicts', 'supporting_evidence_ids', 'opposing_evidence_ids'):
            values = getattr(self, name)
            if not isinstance(values, tuple):
                raise ValueError(f'{name} must be an ordered sequence')
            for value in values:
                identifier(value, name)


@dataclass(frozen=True)
class EvidenceBundle:
    scene_id: str
    items: tuple[EvidenceItem, ...] = ()
    provenance: Mapping = field(default_factory=dict)

    def __post_init__(self):
        identifier(self.scene_id, 'scene_id')
        if not isinstance(self.items, (tuple, list)):
            raise ValueError('items must be an ordered sequence of EvidenceItem')
        seen = set()
        for item in self.items:
            if not isinstance(item, EvidenceItem):
                raise ValueError('items must contain EvidenceItem instances')
            if item.scene_id != self.scene_id:
                raise ValueError('evidence scene_id must match bundle scene_id')
            if item.evidence_id in seen:
                raise ValueError(f'duplicate evidence_id: {item.evidence_id}')
            seen.add(item.evidence_id)
        object.__setattr__(self, 'items', tuple(sorted(self.items, key=lambda x: x.evidence_id)))
        freeze_fields(self, ('provenance',))
