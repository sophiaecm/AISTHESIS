"""Opt-in deterministic execution, never belief formation or production routing."""
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Callable
import json

from .expected_consequences import ExpectedConsequences
from .world_model._structured import freeze, identifier
from .world_model.evidence import EvidenceItem, EvidenceBundle, stable_id
from .world_model.hypothesis import Hypothesis
from .world_model.common_evidence_state import CommonEvidenceState, bounded_plain, check_context


def _strings(values):
    if not isinstance(values, (tuple, list)):
        raise ValueError('expected an ordered string sequence')
    for value in values:
        identifier(value, 'identifier')
    return tuple(sorted(set(values)))


def _mapping(value):
    if not isinstance(value, Mapping):
        raise ValueError('expected structured mapping')
    return freeze(bounded_plain(value))


@dataclass(frozen=True)
class OrchestrationContext:
    common_evidence_state: CommonEvidenceState
    configuration: Mapping = field(default_factory=dict)
    available_capabilities: tuple[str, ...] = ()
    execution_id: str = field(init=False)

    def __post_init__(self):
        if not isinstance(self.common_evidence_state, CommonEvidenceState):
            raise ValueError('expected CommonEvidenceState')
        object.__setattr__(self, 'configuration', _mapping(self.configuration))
        object.__setattr__(self, 'available_capabilities', _strings(self.available_capabilities))
        check_context(self.configuration, **self.alignment(), source_time=self.timestamp, frame_bound=True)
        object.__setattr__(self, 'execution_id', stable_id('orchestration-context',
            self.common_evidence_state.to_dict(), bounded_plain(self.configuration), self.available_capabilities))

    @property
    def session_id(self):
        return self.common_evidence_state.session_id

    @property
    def scene_id(self):
        return self.common_evidence_state.scene_id

    @property
    def timestamp(self):
        return self.common_evidence_state.timestamp

    def alignment(self):
        return dict(session_id=self.session_id, scene_id=self.scene_id, timestamp=self.timestamp,
                    coordinate_frame_id=self.common_evidence_state.coordinate_frame_id)


@dataclass(frozen=True)
class ReasonerRegistration:
    reasoner_id: str
    reasoner_family: str
    version: str
    execute: Callable = field(repr=False, compare=False)
    priority: int = 0
    enabled: bool = True
    required_evidence: tuple[str, ...] = ()
    optional_evidence: tuple[str, ...] = ()
    required_evidence_types: tuple[str, ...] = ()
    required_statuses: Mapping = field(default_factory=dict)
    required_capabilities: tuple[str, ...] = ()
    supported_schemas: tuple[str, ...] = ('common-evidence-state-0.1',)
    coordinate_frame_id: str | None = None

    def __post_init__(self):
        for name in ('reasoner_id', 'reasoner_family', 'version', 'coordinate_frame_id'):
            identifier(getattr(self, name), name, optional=name == 'coordinate_frame_id')
        if not callable(self.execute) or type(self.priority) is not int or type(self.enabled) is not bool:
            raise ValueError('registration requires callable, integer priority and boolean enabled')
        for name in ('required_evidence', 'optional_evidence', 'required_evidence_types',
                     'required_capabilities', 'supported_schemas'):
            object.__setattr__(self, name, _strings(getattr(self, name)))
        if not isinstance(self.required_statuses, Mapping):
            raise ValueError('required_statuses must map families to statuses')
        statuses = {}
        for family, values in self.required_statuses.items():
            identifier(family, 'family')
            statuses[family] = _strings(values)
            if not statuses[family]:
                raise ValueError('required status set cannot be empty')
        object.__setattr__(self, 'required_statuses', _mapping(statuses))
        bounded_plain(self.metadata())

    def metadata(self):
        return {name: bounded_plain(getattr(self, name)) for name in self.__dataclass_fields__ if name != 'execute'}


@dataclass(frozen=True)
class ReasonerRegistry:
    registrations: tuple[ReasonerRegistration, ...] = ()

    def __post_init__(self):
        values = tuple(self.registrations)
        if any(not isinstance(r, ReasonerRegistration) for r in values):
            raise ValueError('registry requires ReasonerRegistration')
        if len({r.reasoner_id for r in values}) != len(values):
            raise ValueError('duplicate reasoner ID')
        object.__setattr__(self, 'registrations', tuple(sorted(values, key=lambda r: (r.priority, r.reasoner_id))))

    def register(self, registration):
        """Return a new registry; no process-global registration/discovery."""
        return ReasonerRegistry(self.registrations + (registration,))


@dataclass(frozen=True)
class Eligibility:
    reasoner_id: str
    status: str
    reasons: tuple[str, ...] = ()

    def __post_init__(self):
        identifier(self.reasoner_id, 'reasoner_id')
        if self.status not in ('eligible', 'disabled', 'unsupported_input',
                               'incompatible_context', 'missing_required_evidence'):
            raise ValueError('invalid eligibility status')
        object.__setattr__(self, 'reasons', _strings(self.reasons))


@dataclass(frozen=True)
class ReasonerOutput:
    structured_output: Mapping = field(default_factory=dict)
    evidence_candidates: tuple[EvidenceItem, ...] = ()
    hypotheses: tuple[Hypothesis, ...] = ()
    uncertainty: Mapping = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, 'structured_output', _mapping(self.structured_output))
        object.__setattr__(self, 'uncertainty', _mapping(self.uncertainty))
        object.__setattr__(self, 'notes', _strings(self.notes))
        for name, kind in (('evidence_candidates', EvidenceItem), ('hypotheses', Hypothesis)):
            values = tuple(getattr(self, name))
            if any(not isinstance(v, kind) for v in values):
                raise ValueError('output requires existing evidence/hypothesis contracts')
            id_field = 'evidence_id' if kind is EvidenceItem else 'hypothesis_id'
            if len({getattr(v, id_field) for v in values}) != len(values):
                raise ValueError('duplicate output identity')
            object.__setattr__(self, name, tuple(sorted(values, key=lambda v: getattr(v, id_field))))
        bounded_plain(self)

    @classmethod
    def from_expected_consequences(cls, output):
        if not isinstance(output, ExpectedConsequences):
            raise ValueError('expected existing ExpectedConsequences')
        return cls(structured_output={'output_type': 'ExpectedConsequences',
                                     'epistemic_role': 'expected', 'predictions': output.predictions})


@dataclass(frozen=True)
class ReasonerResult:
    reasoner_id: str
    execution_status: str
    eligibility: Eligibility
    provenance: Mapping
    output: ReasonerOutput | None = None
    error: Mapping = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.eligibility, Eligibility) or self.reasoner_id != self.eligibility.reasoner_id:
            raise ValueError('result must match eligibility identity')
        if self.execution_status not in ('executed', 'skipped', 'failed'):
            raise ValueError('invalid execution status')
        if (self.output is not None) != (self.execution_status == 'executed'):
            raise ValueError('only executed reasoners may carry outputs')
        if self.output is not None and not isinstance(self.output, ReasonerOutput):
            raise ValueError('result requires ReasonerOutput')
        object.__setattr__(self, 'provenance', _mapping(self.provenance))
        object.__setattr__(self, 'error', _mapping(self.error))


@dataclass(frozen=True)
class OrchestrationResult:
    execution_id: str
    plan: tuple[Eligibility, ...]
    results: tuple[ReasonerResult, ...]

    def __post_init__(self):
        object.__setattr__(self, 'plan', tuple(self.plan))
        object.__setattr__(self, 'results', tuple(self.results))
        if any(not isinstance(p, Eligibility) for p in self.plan) or any(not isinstance(r, ReasonerResult) for r in self.results):
            raise ValueError('invalid plan/result types')
        if tuple(r.eligibility for r in self.results) != self.plan:
            raise ValueError('results must match execution plan')
        bounded_plain(self)

    def to_dict(self):
        return bounded_plain(self)

    def to_json(self):
        return json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'), allow_nan=False)


@dataclass(frozen=True)
class ReasonerOrchestrator:
    registry: ReasonerRegistry

    def __post_init__(self):
        if not isinstance(self.registry, ReasonerRegistry):
            raise ValueError('expected ReasonerRegistry')

    def plan(self, context):
        if not isinstance(context, OrchestrationContext):
            raise ValueError('expected OrchestrationContext')
        state = context.common_evidence_state
        available = tuple(e for e in state.evidence_items if e.epistemic_status != 'unavailable')
        families = {e.source_type.value for e in available}
        categories = {e.evidence_type for e in available}
        plan = []
        for r in self.registry.registrations:
            reasons = ()
            if not r.enabled:
                status = 'disabled'
            elif state.schema_version not in r.supported_schemas:
                status = 'unsupported_input'
            elif not set(r.required_capabilities) <= set(context.available_capabilities):
                status = 'unsupported_input'
                reasons = tuple(sorted(set(r.required_capabilities) - set(context.available_capabilities)))
            elif r.coordinate_frame_id is not None and r.coordinate_frame_id != state.coordinate_frame_id:
                status = 'incompatible_context'
            else:
                missing = [f'family:{f}' for f in r.required_evidence if f not in families]
                missing += [f'category:{c}' for c in r.required_evidence_types if c not in categories]
                missing += [f'status:{f}' for f, statuses in r.required_statuses.items()
                            if not any(e.source_type.value == f and e.epistemic_status in statuses for e in available)]
                status = 'missing_required_evidence' if missing else 'eligible'
                reasons = tuple(sorted(missing))
            plan.append(Eligibility(r.reasoner_id, status, reasons))
        return tuple(plan)

    def run(self, context):
        plan = self.plan(context)
        results = []
        refs = tuple(e.evidence_id for e in context.common_evidence_state.evidence_items)
        for registration, eligibility in zip(self.registry.registrations, plan):
            provenance = {'producer_reasoner_id': registration.reasoner_id,
                'registration': registration.metadata(), 'context_reference': context.execution_id,
                'session_id': context.session_id, 'scene_id': context.scene_id,
                'timestamp': context.timestamp, 'source_evidence_ids': refs,
                'reference_semantics': 'exposed_input_inventory_not_evidence_support',
                'output_role': 'reasoner_output_not_observation'}
            if eligibility.status != 'eligible':
                results.append(ReasonerResult(registration.reasoner_id, 'skipped', eligibility, provenance))
                continue
            try:
                output = registration.execute(context)
                if not isinstance(output, ReasonerOutput):
                    raise ValueError('callable must return ReasonerOutput')
                data = bounded_plain(output)
                check_context(data, **context.alignment(), source_time=context.timestamp, frame_bound=True)
                for candidate in output.evidence_candidates + output.hypotheses:
                    p = candidate.provenance
                    if p.get('producer_reasoner_id') != registration.reasoner_id:
                        raise ValueError('candidate must identify its producer')
                    if any(p.get(key) != value for key, value in (
                        ('session_id', context.session_id), ('scene_id', context.scene_id), ('timestamp', context.timestamp))):
                        raise ValueError('candidate requires matching source context')
                    derived = p.get('derived_from')
                    if not isinstance(derived, tuple) or not set(derived) <= set(refs):
                        raise ValueError('candidate requires valid input evidence references')
                    if isinstance(candidate, EvidenceItem):
                        if candidate.epistemic_status is None:
                            raise ValueError('candidate requires explicit epistemic status')
                    elif set(candidate.evidence_for + candidate.evidence_against) - set(refs):
                        raise ValueError('hypothesis references unknown evidence')
                    if isinstance(candidate, Hypothesis) and candidate.target_timestamp != candidate.created_timestamp + candidate.horizon_seconds:
                        raise ValueError('hypothesis target must match creation time plus horizon')
                results.append(ReasonerResult(registration.reasoner_id, 'executed', eligibility, provenance, output))
            except Exception as exc:
                # No arbitrary exception messages, reprs, paths, credentials or tracebacks.
                results.append(ReasonerResult(registration.reasoner_id, 'failed', eligibility, provenance,
                    error={'type': type(exc).__name__[:80], 'code': 'execution_or_output_validation_failed'}))
        execution_id = stable_id('orchestration-run', context.execution_id,
                                 [r.metadata() for r in self.registry.registrations])
        return OrchestrationResult(execution_id, plan, tuple(results))
