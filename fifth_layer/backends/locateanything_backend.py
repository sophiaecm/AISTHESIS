"""Injected LocateAnything boundary; no model loader, I/O, or core inference."""
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
import json
import re
from typing import Protocol

from fifth_layer.world_model._structured import freeze, identifier
from fifth_layer.world_model.common_evidence_state import bounded_plain
from fifth_layer.world_model.evidence import stable_id
from fifth_layer.world_model.spatial_grounding import GroundingRequest, RawGroundingObservation, _check

VERSION = 'locateanything-backend-v0.1'
BACKEND = 'LocateAnything'
MODEL = 'nvidia/LocateAnything-3B'
PARSER = 'explicit-normalized-box-point-v0.1'
LIMITS = ('external_backend_output_unverified', 'grounding_not_physical_truth',
          'backend_label_not_verified_identity', 'backend_score_not_aisthesis_confidence',
          'no_candidate_not_physical_absence', 'normalized_2d_not_metric_3d',
          'grounding_not_control', 'real_backend_not_smoke_tested')
POLICY = dict(backend_boundary='external', executable_control=False, hidden_actor_inference=False,
              world_state_mutation=False, backend_output_unverified=True, physical_truth_claim=False,
              object_identity_verified=False, model_output_not_observation_truth=True,
              backend_score_promoted=False, Bayesian_feedback=False, ExperienceLearning_feedback=False,
              branch_selection=False, control=False, task_level_validation=False)


def _strings(value):
    if not isinstance(value, (tuple, list)):
        raise ValueError('ordered string sequence required')
    for item in value:
        identifier(item, 'reference')
    return tuple(sorted(set(value)))


def _task(outputs):
    return {('box',): 'grounding', ('point',): 'pointing', ('box', 'point'): 'grounding_and_pointing'}.get(outputs)


class _Record:
    def to_dict(self):
        return bounded_plain(self)

    def to_json(self):
        return json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'), allow_nan=False)

    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if f.init and f.name.endswith('_id'):
                identifier(value, f.name)
            elif f.name in ('uncertainty', 'requested_output_types'):
                object.__setattr__(self, f.name, _strings(value))
            elif f.name in ('provenance', 'backend_metadata', 'structured_payload'):
                if value is None and f.name == 'structured_payload':
                    continue
                if not isinstance(value, Mapping):
                    raise ValueError('bounded mapping required')
                object.__setattr__(self, f.name, freeze(bounded_plain(value)))
        if self.backend_name != BACKEND or self.backend_model != MODEL:
            raise ValueError('unexpected backend identity')
        identity = next(f.name for f in fields(self) if not f.init and f.name.endswith('_id'))
        data = self.to_dict()
        data.pop(identity)
        object.__setattr__(self, identity, stable_id(identity, VERSION, data))


@dataclass(frozen=True)
class LocateAnythingInvocation(_Record):
    request_id: str
    frame_id: str
    image_reference: str
    query: str
    task_type: str
    requested_output_types: tuple
    backend_name: str = BACKEND
    backend_model: str = MODEL
    provenance: Mapping = field(default_factory=dict)
    invocation_id: str = field(default='', init=False)
    schema_version: str = field(default=VERSION, init=False)

    def __post_init__(self):
        identifier(self.image_reference, 'image_reference')
        identifier(self.query, 'explicit caller query')
        outputs = _strings(self.requested_output_types)
        if _task(outputs) is None or self.task_type != _task(outputs):
            raise ValueError('task/output mismatch')
        if self.provenance.get('query_source') != 'caller_provided':
            raise ValueError('explicit caller query provenance required')
        object.__setattr__(self, 'provenance', dict(self.provenance, **POLICY))
        super().__post_init__()


@dataclass(frozen=True)
class LocateAnythingRawResponse(_Record):
    invocation_id: str
    request_id: str
    frame_id: str
    response_text: str | None = None
    structured_payload: Mapping | None = None
    backend_name: str = BACKEND
    backend_model: str = MODEL
    backend_version: str = 'unspecified'
    backend_metadata: Mapping = field(default_factory=dict)
    uncertainty: tuple = ()
    provenance: Mapping = field(default_factory=dict)
    response_id: str = field(default='', init=False)
    schema_version: str = field(default=VERSION, init=False)

    def __post_init__(self):
        if self.response_text is not None and not isinstance(self.response_text, str):
            raise ValueError('bounded response text required')
        identifier(self.backend_version, 'backend_version')
        if self.structured_payload is not None:
            payload = bounded_plain(self.structured_payload)
            if not isinstance(payload, dict):
                raise ValueError('structured response mapping required')
            if 'results' in payload:
                if not isinstance(payload['results'], list):
                    raise ValueError('ordered results required')
                payload['results'] = sorted(payload['results'], key=lambda r: json.dumps(r, sort_keys=True))
            object.__setattr__(self, 'structured_payload', payload)
        object.__setattr__(self, 'uncertainty', tuple(sorted(set((*_strings(self.uncertainty), *LIMITS)))))
        object.__setattr__(self, 'provenance', dict(self.provenance, **POLICY))
        super().__post_init__()


class LocateAnythingRuntime(Protocol):
    def infer(self, invocation: LocateAnythingInvocation) -> LocateAnythingRawResponse:
        """External runtime owns loading and inference; no concrete implementation here."""
        ...


def _invocation_for_request(request, invocation):
    _check(request, GroundingRequest)
    _check(invocation, LocateAnythingInvocation, VERSION)
    if (invocation.request_id, invocation.frame_id, invocation.requested_output_types) != (
            request.request_id, request.frame.frame_id, request.requested_output_types):
        raise ValueError('invocation/request lineage mismatch')


def _response_for_invocation(invocation, response):
    _check(response, LocateAnythingRawResponse, VERSION)
    for name, expected in (('invocation_id', invocation.invocation_id), ('request_id', invocation.request_id),
                           ('frame_id', invocation.frame_id), ('backend_name', invocation.backend_name),
                           ('backend_model', invocation.backend_model)):
        if getattr(response, name) != expected:
            raise ValueError('response lineage mismatch: ' + name)


def _text_records(text):
    if text == '<no_candidate/>':
        return []
    if not text:
        raise ValueError('missing output is not an explicit no-candidate response')
    records = []
    rest = text
    # Every angle-delimited fragment must belong to a complete supported record.
    while '<box>' in rest:
        prefix, remainder = rest.split('<box>', 1)
        if '<' in prefix or '>' in prefix or '</box>' not in remainder:
            raise ValueError('malformed text delimiters')
        body, rest = remainder.split('</box>', 1)
        if re.fullmatch(r'(?:<[0-9]{1,4}>){2}|(?:<[0-9]{1,4}>){4}', body) is None:
            raise ValueError('unsupported coordinate token grammar')
        values = tuple(int(token) for token in body[1:-1].split('><'))
        records.append(dict(type='box' if len(values) == 4 else 'point', coordinates=values))
    if '<' in rest or '>' in rest or not records:
        raise ValueError('no supported explicit grounding record or malformed delimiter')
    return records


class LocateAnythingOutputParser:
    def parse(self, request, invocation, response):
        _invocation_for_request(request, invocation)
        _response_for_invocation(invocation, response)
        payload = response.structured_payload
        if payload is None:
            records = _text_records(response.response_text)
        else:
            if set(payload) != {'status', 'results'} or payload['status'] not in ('completed', 'no_candidate'):
                raise ValueError('structured grammar requires status and results')
            records = payload['results']
            if payload['status'] == 'no_candidate' and records:
                raise ValueError('no-candidate response cannot contain results')
            if payload['status'] == 'completed' and not records:
                raise ValueError('empty completed payload needs explicit no_candidate status')
            if response.response_text and ('<' in response.response_text or '>' in response.response_text):
                raise ValueError('two coordinate-bearing representations are not accepted together')
        observations = {}
        for record in records:
            if not isinstance(record, Mapping) or not {'type', 'coordinates'} <= record.keys() or not set(record) <= {
                    'type', 'coordinates', 'label', 'score', 'uncertainty', 'raw_reference'}:
                raise ValueError('unsupported structured result fields')
            if record['type'] not in request.requested_output_types:
                raise ValueError('unrequested output type')
            uncertainty = tuple(sorted(set((*request.uncertainty, *response.uncertainty,
                                            *_strings(record.get('uncertainty', ())), *LIMITS))))
            observation = RawGroundingObservation(request.request_id, request.frame.frame_id,
                response.backend_name, response.backend_version, record['type'], record['coordinates'],
                'normalized_0_1000', backend_label=record.get('label'), backend_score=record.get('score'),
                raw_reference=record.get('raw_reference', response.response_id), uncertainty=uncertainty,
                provenance=dict(POLICY, parsed_from_external_backend=True, coordinate_space='normalized_0_1000',
                    backend_model=response.backend_model, invocation_id=invocation.invocation_id,
                    response_id=response.response_id, query_source='caller_provided', query=invocation.query,
                    image_reference=invocation.image_reference, parser_policy=PARSER,
                    response_uncertainty=response.uncertainty, backend_metadata=response.backend_metadata,
                    response_provenance=response.provenance))
            observations[observation.observation_id] = observation
        return tuple(sorted(observations.values(), key=lambda o: o.observation_id))


class LocateAnythingBackendBridge:
    def create_invocation(self, request, image_reference, query=None):
        _check(request, GroundingRequest)
        # Frozen Step 26B only supports semantic_query_status=unavailable.
        identifier(query, 'explicit caller query required')
        return LocateAnythingInvocation(request.request_id, request.frame.frame_id, image_reference, query,
            _task(request.requested_output_types), request.requested_output_types,
            provenance=dict(POLICY, query_source='caller_provided', source_plan_id=request.source_plan_id,
                            source_target_id=request.source_target_id))

    def run(self, invocation, runtime):
        _check(invocation, LocateAnythingInvocation, VERSION)
        infer = getattr(runtime, 'infer', None)
        if not callable(infer):
            raise ValueError('injected runtime infer method required')
        response = infer(invocation)
        _response_for_invocation(invocation, response)
        return response

    def parse(self, request, invocation, response):
        return LocateAnythingOutputParser().parse(request, invocation, response)

    def ground(self, request, image_reference, runtime, query=None):
        invocation = self.create_invocation(request, image_reference, query)
        response = self.run(invocation, runtime)
        return self.parse(request, invocation, response)
