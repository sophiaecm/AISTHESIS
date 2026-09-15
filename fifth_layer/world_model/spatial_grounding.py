"""Backend-neutral localization evidence boundary; no backend or control execution."""
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
import json

from ._structured import freeze, identifier, number
from .common_evidence_state import bounded_plain
from .evidence import stable_id
from .active_perception import ObservationRequestPlan, SCOPES

VERSION = 'spatial-grounding-v0.1'
OUTPUTS = ('box', 'point')
SPACES = ('normalized_0_1000', 'normalized_0_1', 'pixel')
LIMITS = ('grounding_not_physical_truth', 'pixel_location_not_world_location',
          'backend_label_not_verified_identity', 'backend_score_not_aisthesis_confidence',
          'no_candidate_not_physical_absence', 'grounding_not_control',
          'external_backend_not_validated_by_step26b', 'continuous_image_edge_coordinates')
POLICY = dict(advisory_only=True, executable_control=False, physical_truth_claim=False,
              hidden_actor_inference=False, object_identity_verified=False, Bayesian_feedback=False,
              ExperienceLearning_feedback=False, branch_selection=False, risk_update=False,
              world_state_mutation=False, backend_score_promoted=False, task_level_validation=False)


def _strings(values):
    if not isinstance(values, (tuple, list)):
        raise ValueError('ordered string sequence required')
    for value in values:
        identifier(value, 'reference')
    return tuple(sorted(set(values)))


def _dimensions(width, height):
    if any(type(v) is not int or v <= 0 for v in (width, height)):
        raise ValueError('positive integer frame dimensions required')
    number(width, 'width')
    number(height, 'height')


def _coordinates(kind, coordinates, xmax=None, ymax=None):
    if kind not in OUTPUTS or not isinstance(coordinates, (tuple, list)) or len(coordinates) != (4 if kind == 'box' else 2):
        raise ValueError('unsupported output type or coordinate arity')
    for i, value in enumerate(coordinates):
        number(value, 'coordinate', nonnegative=True)
        bound = xmax if i % 2 == 0 else ymax
        if bound is not None and value > bound:
            raise ValueError('coordinate outside bounds')
    if kind == 'box' and (coordinates[0] > coordinates[2] or coordinates[1] > coordinates[3]):
        raise ValueError('reversed box')
    return tuple(coordinates)


def _check(record, cls, version=VERSION):
    if not isinstance(record, cls) or record.schema_version != version:
        raise ValueError('typed source with expected schema required')
    if replace(record).to_dict() != record.to_dict():
        raise ValueError('inconsistent content-derived identity or source structure')


class _Record:
    def to_dict(self):
        return bounded_plain(self)

    def to_json(self):
        return json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'), allow_nan=False)

    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name.endswith('_ids') or f.name in ('relation_types', 'uncertainty', 'requested_output_types'):
                object.__setattr__(self, f.name, _strings(value))
            elif f.name in ('provenance', 'symbolic_query'):
                if not isinstance(value, Mapping):
                    raise ValueError('mapping metadata required')
                object.__setattr__(self, f.name, freeze(bounded_plain(value)))
            elif f.name == 'backend_metadata':
                if not isinstance(value, (tuple, list)) or any(not isinstance(v, Mapping) for v in value):
                    raise ValueError('ordered backend metadata mappings required')
                object.__setattr__(self, f.name, tuple(sorted((freeze(bounded_plain(v)) for v in value),
                    key=lambda v: json.dumps(bounded_plain(v), sort_keys=True))))
            elif f.init and f.name.endswith('_id'):
                identifier(value, f.name)
        identity = next(f.name for f in fields(self) if not f.init and f.name.endswith('_id'))
        data = self.to_dict()
        data.pop(identity)
        object.__setattr__(self, identity, stable_id(identity, VERSION, data))


@dataclass(frozen=True)
class GroundingFrame(_Record):
    image_id: str
    width: int
    height: int
    timestamp: float
    session_id: str
    coordinate_frame_id: str
    scene_id: str
    provenance: Mapping = field(default_factory=dict)
    frame_id: str = field(default='', init=False)
    schema_version: str = field(default=VERSION, init=False)

    def __post_init__(self):
        _dimensions(self.width, self.height)
        number(self.timestamp, 'timestamp', nonnegative=True)
        super().__post_init__()


@dataclass(frozen=True)
class GroundingRequest(_Record):
    source_plan_id: str
    source_target_id: str
    target_scope: str
    object_ids: tuple
    relation_types: tuple
    symbolic_query: Mapping
    frame: GroundingFrame
    requested_output_types: tuple
    uncertainty: tuple
    provenance: Mapping
    request_id: str = field(default='', init=False)
    schema_version: str = field(default=VERSION, init=False)

    def __post_init__(self):
        _check(self.frame, GroundingFrame)
        if self.target_scope not in SCOPES or not self.object_ids:
            raise ValueError('explicit supported symbolic target required')
        if (self.target_scope == 'relation') != bool(self.relation_types):
            raise ValueError('relation scope/type mismatch')
        if self.target_scope == 'object' and len(set(self.object_ids)) != 1:
            raise ValueError('single object scope required')
        if not self.requested_output_types or not set(self.requested_output_types) <= set(OUTPUTS):
            raise ValueError('only box/point requests supported')
        expected = dict(object_ids=_strings(self.object_ids), relation_types=_strings(self.relation_types),
                        target_scope=self.target_scope, semantic_query_status='unavailable')
        if bounded_plain(self.symbolic_query) != bounded_plain(expected):
            raise ValueError('only exact symbolic queries supported; no inferred semantics')
        super().__post_init__()


@dataclass(frozen=True)
class RawGroundingObservation(_Record):
    request_id: str
    frame_id: str
    backend_name: str
    backend_version: str
    output_type: str
    coordinates: tuple
    coordinate_space: str
    backend_label: str | None = None
    backend_score: float | None = None
    raw_reference: str | None = None
    uncertainty: tuple = ()
    provenance: Mapping = field(default_factory=dict)
    observation_id: str = field(default='', init=False)
    schema_version: str = field(default=VERSION, init=False)

    def __post_init__(self):
        for name in ('backend_name', 'backend_version'):
            identifier(getattr(self, name), name)
        for name in ('backend_label', 'raw_reference'):
            identifier(getattr(self, name), name, optional=True)
        if self.backend_score is not None:
            number(self.backend_score, 'backend_score')
        if self.coordinate_space not in SPACES:
            raise ValueError('unsupported coordinate space')
        maximum = {'normalized_0_1000': 1000, 'normalized_0_1': 1, 'pixel': None}[self.coordinate_space]
        object.__setattr__(self, 'coordinates', _coordinates(self.output_type, self.coordinates, maximum, maximum))
        super().__post_init__()


@dataclass(frozen=True)
class GroundedRegion(_Record):
    output_type: str
    pixel_coordinates: tuple
    normalized_coordinates: tuple
    frame_width: int
    frame_height: int
    source_observation_id: str
    frame_id: str
    uncertainty: tuple
    provenance: Mapping
    region_id: str = field(default='', init=False)
    schema_version: str = field(default=VERSION, init=False)

    def __post_init__(self):
        _dimensions(self.frame_width, self.frame_height)
        pixels = _coordinates(self.output_type, self.pixel_coordinates, self.frame_width, self.frame_height)
        normalized = _coordinates(self.output_type, self.normalized_coordinates, 1, 1)
        expected = tuple(v / (self.frame_width if i % 2 == 0 else self.frame_height) for i, v in enumerate(pixels))
        if expected != normalized:
            raise ValueError('inconsistent canonical normalized coordinates')
        object.__setattr__(self, 'pixel_coordinates', pixels)
        object.__setattr__(self, 'normalized_coordinates', normalized)
        super().__post_init__()


@dataclass(frozen=True)
class GroundedObservationTarget(_Record):
    source_plan_id: str
    source_target_id: str
    source_request_id: str
    target_scope: str
    object_ids: tuple
    relation_types: tuple
    frame: GroundingFrame
    regions: tuple
    grounding_status: str
    backend_metadata: tuple
    uncertainty: tuple
    provenance: Mapping
    grounded_target_id: str = field(default='', init=False)
    schema_version: str = field(default=VERSION, init=False)

    def __post_init__(self):
        _check(self.frame, GroundingFrame)
        if self.target_scope not in SCOPES or not self.object_ids or (self.target_scope == 'relation') != bool(self.relation_types):
            raise ValueError('invalid symbolic target scope')
        if self.target_scope == 'object' and len(set(self.object_ids)) != 1:
            raise ValueError('single object scope required')
        if not isinstance(self.regions, (tuple, list)):
            raise ValueError('ordered regions required')
        for region in self.regions:
            _check(region, GroundedRegion)
            if (region.frame_id, region.frame_width, region.frame_height) != (self.frame.frame_id, self.frame.width, self.frame.height):
                raise ValueError('region frame mismatch')
        if len({r.region_id for r in self.regions}) != len(self.regions):
            raise ValueError('duplicate region')
        object.__setattr__(self, 'regions', tuple(sorted(self.regions, key=lambda r: r.region_id)))
        expected = 'multiple_candidates' if len(self.regions) > 1 else 'grounded_candidate' if self.regions else 'no_candidate'
        if self.grounding_status != expected and not (not self.regions and self.grounding_status in ('unavailable', 'indeterminate')):
            raise ValueError('grounding status disagrees with region count')
        super().__post_init__()
        observation_ids = [m.get('source_observation_id') for m in self.backend_metadata]
        if len(set(observation_ids)) != len(observation_ids) or set(observation_ids) != {r.source_observation_id for r in self.regions}:
            raise ValueError('backend metadata must match region observation lineage')


class SpatialGroundingAdapter:
    def create_request(self, plan, target_id, frame, requested_output_types=('box', 'point')):
        _check(plan, ObservationRequestPlan, 'active-perception-v0.1')
        for record in (*plan.cues, *plan.targets):
            _check(record, type(record), 'active-perception-v0.1')
        _check(frame, GroundingFrame)
        identifier(target_id, 'target_id')
        for name in ('scene_id', 'timestamp', 'session_id', 'coordinate_frame_id'):
            if getattr(plan, name) != getattr(frame, name):
                raise ValueError('frame/plan alignment mismatch: ' + name)
        target = next((t for t in plan.targets if t.target_id == target_id), None)
        if target is None:
            raise ValueError('target ID not in supplied plan')
        query = dict(target_scope=target.target_scope, object_ids=target.object_ids,
                     relation_types=target.relation_types, semantic_query_status='unavailable')
        uncertainty = tuple(sorted(set((*plan.uncertainty, *target.uncertainty, *LIMITS, 'semantic_query_unavailable'))))
        provenance = dict(POLICY, external_grounding_evidence=False,
            source_topological_state_id=plan.source_topological_state_id,
            source_complex_state_id=plan.source_complex_state_id,
            source_physical_state_id=plan.source_physical_state_id,
            source_cue_ids=target.cue_ids, source_ids=target.source_ids,
            source_priority_band=target.priority_band, source_rationale_code=target.rationale_code,
            source_plan_uncertainty=plan.uncertainty, source_target_uncertainty=target.uncertainty,
            semantic_query_status='unavailable', semantic_resolution='deferred')
        return GroundingRequest(plan.plan_id, target.target_id, target.target_scope, target.object_ids,
                                target.relation_types, query, frame, requested_output_types, uncertainty, provenance)

    def integrate_observations(self, request, observations, *, response_status='completed'):
        """Caller explicitly supplies a complete result, unavailable, or indeterminate response.

        An explicit empty completed sequence means no candidates. No backend is called.
        Malformed mixed responses fail as a whole; exact duplicate records coalesce.
        """
        _check(request, GroundingRequest)
        if response_status not in ('completed', 'unavailable', 'indeterminate'):
            raise ValueError('unsupported backend response status')
        if not isinstance(observations, (tuple, list)):
            raise ValueError('explicit observation sequence required')
        if response_status != 'completed' and observations:
            raise ValueError('non-completed response cannot contain localization candidates')
        unique = {}
        for observation in observations:
            _check(observation, RawGroundingObservation)
            if observation.request_id != request.request_id or observation.frame_id != request.frame.frame_id:
                raise ValueError('observation request/frame mismatch')
            if observation.output_type not in request.requested_output_types:
                raise ValueError('unrequested output type')
            unique[observation.observation_id] = observation
        regions, metadata = [], []
        uncertainty = set(request.uncertainty) | set(LIMITS)
        for observation in sorted(unique.values(), key=lambda o: o.observation_id):
            frame = request.frame
            coordinates = observation.coordinates
            if observation.coordinate_space == 'pixel':
                pixels = _coordinates(observation.output_type, coordinates, frame.width, frame.height)
            else:
                divisor = 1000 if observation.coordinate_space == 'normalized_0_1000' else 1
                pixels = tuple(v / divisor * (frame.width if i % 2 == 0 else frame.height) for i, v in enumerate(coordinates))
            normalized = tuple(v / (frame.width if i % 2 == 0 else frame.height) for i, v in enumerate(pixels))
            flags = tuple(sorted(set((*request.uncertainty, *observation.uncertainty, *LIMITS))))
            regions.append(GroundedRegion(observation.output_type, pixels, normalized, frame.width, frame.height,
                observation.observation_id, frame.frame_id, flags,
                dict(POLICY, external_grounding_evidence=True, source_coordinate_space=observation.coordinate_space,
                     source_coordinates=coordinates, conversion='continuous_edges_no_rounding_or_clamping')))
            metadata.append(dict(source_observation_id=observation.observation_id, backend_name=observation.backend_name,
                backend_version=observation.backend_version, backend_label=observation.backend_label,
                backend_score=observation.backend_score, raw_reference=observation.raw_reference,
                uncertainty=observation.uncertainty, backend_provenance=observation.provenance))
            uncertainty.update(flags)
        status = response_status if response_status != 'completed' else (
            'multiple_candidates' if len(regions) > 1 else 'grounded_candidate' if regions else 'no_candidate')
        if status == 'multiple_candidates':
            uncertainty.add('multiple_candidate_ambiguity')
        if status in ('unavailable', 'indeterminate'):
            uncertainty.add('backend_response_' + status)
        provenance = dict(request.provenance, **{'external_grounding_evidence': response_status == 'completed'},
                          response_status=response_status, source_frame_id=request.frame.frame_id,
                          source_image_id=request.frame.image_id, winner_selection=False,
                          duplicate_policy='identical_observation_ids_coalesce')
        provenance.update(POLICY)
        return GroundedObservationTarget(request.source_plan_id, request.source_target_id, request.request_id,
            request.target_scope, request.object_ids, request.relation_types, request.frame, tuple(regions),
            status, tuple(metadata), tuple(sorted(uncertainty)), provenance)
