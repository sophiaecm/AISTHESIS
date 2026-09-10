"""Detached, immutable summaries; deliberately no array/tensor dependencies."""
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from math import isfinite
from types import MappingProxyType


SCHEMA_VERSION = '0.1'
_OMIT = object()
_BUFFER_KEYS = frozenset({
    'image', 'frame', 'raw_image', 'raw_frame', 'video_frame', 'pixels',
    'image_data', 'frame_data', 'image_buffer', 'encoded_image', 'image_bytes',
    'frame_bytes', 'buffer', 'tensor', 'state_json', 'motion_json',
})


def number(value, name, *, unit=False, nonnegative=False):
    if type(value) not in (int, float) or not isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    if unit and not 0 <= value <= 1:
        raise ValueError(f'{name} must be between 0.0 and 1.0')
    if nonnegative and value < 0:
        raise ValueError(f'{name} must be non-negative')


def identifier(value, name, optional=False):
    if optional and value is None:
        return
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a non-empty string')


def geometry(value, name, size, xywh=False):
    if not isinstance(value, (tuple, list)) or len(value) != size:
        raise ValueError(f'{name} must contain {size} finite coordinates')
    for coordinate in value:
        number(coordinate, name)
    if size == 4:
        valid = (value[2] >= 0 and value[3] >= 0 if xywh else
                 value[2] >= value[0] and value[3] >= value[1])
        if not valid:
            raise ValueError(f'{name} must have non-negative width and height')


def freeze(value, path='summary', *, sanitize=False):
    """Copy lists/maps/sets to tuples/read-only maps/frozensets.

    Only plain scalar and structured data is accepted. Adapter mode drops
    unsupported leaves and buffer-named fields; direct construction rejects
    them. Bytes, ndarrays, PIL objects and tensors never cross this boundary.
    Large strings are rejected too, including encoded visual payloads.
    Cyclic structures are rejected with a descriptive error.
    """
    def visit(item, location, ancestors):
        if isinstance(item, Enum):
            item = item.value
        if item is None or type(item) in (bool, int, float, str):
            if isinstance(item, str) and len(item) > 16384:
                if sanitize:
                    return _OMIT
                raise ValueError(f'{location} exceeds the summary string limit')
            if type(item) in (int, float):
                number(item, location)
            return item
        if id(item) in ancestors:
            raise ValueError(f'{location} contains a cyclic structure')
        ancestors = ancestors | {id(item)}
        if is_dataclass(item) and not isinstance(item, type):
            item = {f.name: getattr(item, f.name) for f in fields(item)}
        if isinstance(item, Mapping):
            result = {}
            for key, child in item.items():
                if type(key) is not str or key.lower() in _BUFFER_KEYS:
                    if sanitize:
                        continue
                    raise ValueError(f'{location} contains a buffer field or non-string key')
                copied = visit(child, f'{location}.{key}', ancestors)
                if copied is not _OMIT:
                    result[key] = copied
            return MappingProxyType(result)
        if isinstance(item, (list, tuple, set, frozenset)):
            copied = [visit(child, location, ancestors) for child in item]
            copied = [child for child in copied if child is not _OMIT]
            return frozenset(copied) if isinstance(item, (set, frozenset)) else tuple(copied)
        if sanitize:
            return _OMIT
        raise ValueError(f'{location} contains unsupported data: {type(item).__name__}')

    result = visit(value, path, set())
    return None if result is _OMIT else result


def validate_summary(value, path='summary'):
    """Validate known repository geometry/probability fields, preserving units.

    position_uncertainty is intentionally excluded: existing tracks use pixels.
    No scoring or interpretation of arbitrary evidence fields happens here.
    """
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = f'{path}.{key}'
            if key == 'source_field_mapping':
                if not isinstance(child, Mapping) or any(type(v) is not str for v in child.values()):
                    raise ValueError(f'{name} must map destination names to source field names')
                continue
            if child is not None:
                if key in {'confidence', 'prior_probability', 'posterior_probability',
                           'uncertainty', 'fused_uncertainty', 'auditory_uncertainty'} or key.endswith('_confidence'):
                    number(child, name, unit=True)
                elif key in {'center', 'predicted_center', 'origin_center', 'observed_center'}:
                    geometry(child, name, 2)
                elif key in {'bbox', 'box_xyxy', 'predicted_bbox', 'origin_bbox', 'observed_bbox'}:
                    geometry(child, name, 4)
                elif key == 'box':
                    geometry(child, name, 4, xywh=True)
                elif key == 'timestamp' or key.endswith('_timestamp') or key == 'horizon_seconds':
                    number(child, name, nonnegative=True)
            validate_summary(child, name)
    elif isinstance(value, (tuple, frozenset)):
        for child in value:
            validate_summary(child, path)


def freeze_fields(instance, names):
    for name in names:
        value = freeze(getattr(instance, name), name)
        if name == 'provenance' and not isinstance(value, Mapping):
            raise ValueError('provenance must be a structured mapping')
        validate_summary(value, name)
        object.__setattr__(instance, name, value)


def timestamps(instance, names, *, optional=()):
    for name in names:
        value = getattr(instance, name)
        if value is not None or name not in optional:
            number(value, name, nonnegative=True)
