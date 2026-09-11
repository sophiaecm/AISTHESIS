"""Immutable image-plane physical summaries, independent of prediction systems."""
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
import json

from ._structured import freeze, identifier, timestamps


def plain(value):
    if is_dataclass(value):
        return {f.name: plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {k: plain(v) for k, v in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


@dataclass(frozen=True)
class PhysicalAttribute:
    value: object = None
    status: str = 'unknown'
    derived_from: tuple[str, ...] = ()
    rule: str | None = None
    units: str | None = None

    def __post_init__(self):
        if self.status not in ('observed', 'estimated', 'possible', 'unknown', 'unavailable'):
            raise ValueError('invalid physical attribute status')
        if (self.value is None) != (self.status in ('unknown', 'unavailable')):
            raise ValueError('unknown/unavailable attributes require None; known attributes require a value')
        if self.value is not None and not self.derived_from:
            raise ValueError('physical values require provenance')
        if self.status in ('estimated', 'possible') and not self.rule:
            raise ValueError('derived values require a rule')
        object.__setattr__(self, 'value', freeze(self.value))
        refs = tuple(sorted(set(self.derived_from)))
        for ref in refs:
            identifier(ref, 'derived_from')
        object.__setattr__(self, 'derived_from', refs)


@dataclass(frozen=True)
class PhysicalObjectState:
    object_id: str
    timestamp: float | None
    attributes: Mapping[str, PhysicalAttribute] = field(default_factory=dict)
    provenance: Mapping = field(default_factory=dict)

    def __post_init__(self):
        identifier(self.object_id, 'object_id')
        timestamps(self, ('timestamp',), optional=('timestamp',))
        if any(not isinstance(v, PhysicalAttribute) for v in self.attributes.values()):
            raise ValueError('attributes must contain PhysicalAttribute values')
        # Keep typed attributes, detach the mapping itself.
        from types import MappingProxyType
        object.__setattr__(self, 'attributes', MappingProxyType(dict(sorted(self.attributes.items()))))
        object.__setattr__(self, 'provenance', freeze(self.provenance))


@dataclass(frozen=True)
class PhysicalRelation:
    subject_id: str
    relation: str
    object_id: str
    evidence: PhysicalAttribute

    def __post_init__(self):
        for name in ('subject_id', 'object_id'):
            identifier(getattr(self, name), name)
        if self.subject_id == self.object_id:
            raise ValueError('self relation is unsupported')
        if self.relation not in ('left_of', 'right_of', 'above', 'below', 'overlaps',
                                 'contact_possible', 'support_possible'):
            raise ValueError('unsupported physical relation')
        if not isinstance(self.evidence, PhysicalAttribute):
            raise ValueError('relation requires attributed evidence')


@dataclass(frozen=True)
class PhysicalWorldState:
    scene_id: str
    timestamp: float | None
    objects: tuple[PhysicalObjectState, ...] = ()
    relations: tuple[PhysicalRelation, ...] = ()
    evidence_references: tuple[str, ...] = ()
    uncertainty: Mapping = field(default_factory=dict)
    provenance: Mapping = field(default_factory=dict)
    schema_version: str = 'physical-world-0.1'

    def __post_init__(self):
        identifier(self.scene_id, 'scene_id')
        timestamps(self, ('timestamp',), optional=('timestamp',))
        objects = tuple(sorted(self.objects, key=lambda o: o.object_id))
        ids = {o.object_id for o in objects}
        if len(ids) != len(objects):
            raise ValueError('duplicate physical object identity')
        if any(o.timestamp != self.timestamp for o in objects):
            raise ValueError('object timestamps must match snapshot')
        if any(r.subject_id not in ids or r.object_id not in ids for r in self.relations):
            raise ValueError('relations require currently observed endpoints')
        object.__setattr__(self, 'objects', objects)
        object.__setattr__(self, 'relations', tuple(sorted(self.relations,
            key=lambda r: (r.subject_id, r.relation, r.object_id, json.dumps(plain(r.evidence), sort_keys=True)))))
        object.__setattr__(self, 'evidence_references', tuple(sorted(set(self.evidence_references))))
        for name in ('uncertainty', 'provenance'):
            object.__setattr__(self, name, freeze(getattr(self, name)))

    def to_dict(self):
        return plain(self)

    def to_json(self):
        return json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'), allow_nan=False)
