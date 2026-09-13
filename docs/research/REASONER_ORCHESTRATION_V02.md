# Agent / reasoner orchestration v0.2

Status: **ENGINEERING MILESTONE**.

- IMPLEMENTED
- TESTED (engineering correctness)
- NOT YET TASK-LEVEL EVALUATED
- NOT YET EXPERIMENTALLY SUPPORTED

Orchestration determines execution, not truth.

Reasoner execution is not evidence support.

Agreement between reasoners does not automatically increase
confidence.

Reasoner failure is not evidence contradiction.

This layer is a deterministic execution substrate for future dynamic
reasoning routing.

## Repository fit

BaseReasoner defines WorldState -> ExpectedConsequences -> LatentState ->
FutureState. FifthLayerEngine maintains last_future_state. CompositeReasoner
combines named outputs, and the production AisthesisOrchestrator calls a fixed
set of WorldState reasoners. None supplies CommonEvidenceState registration,
eligibility, or per-call failure envelopes. They remain unchanged.

The opt-in fifth_layer/orchestration.py module accepts lightweight callables;
it does not introduce another inference base class or duplicate evidence or
hypothesis ontology. EvidenceItem, EvidenceBundle, Hypothesis, CommonEvidenceState,
and ExpectedConsequences are reused. No production adapter is automatically
registered. The compatibility test calls the existing PhysicsReasoner and
transports its ExpectedConsequences without rewriting the reasoner.

## Registration and execution

```python
from fifth_layer.orchestration import (
    OrchestrationContext, ReasonerRegistration, ReasonerRegistry,
    ReasonerOutput, ReasonerOrchestrator,
)

def adapter(context):
    # A deterministic callable consuming the frozen input inventory.
    return ReasonerOutput(structured_output={'note': 'synthetic example'})

registry = ReasonerRegistry().register(ReasonerRegistration(
    reasoner_id='example', reasoner_family='custom', version='0.1',
    execute=adapter, priority=10, required_evidence=('physical',),
))
context = OrchestrationContext(common_evidence_state)
plan = ReasonerOrchestrator(registry).plan(context)
result = ReasonerOrchestrator(registry).run(context)
serialized = result.to_json()
```

Registration is explicit and local. register returns a new immutable registry;
it does not modify the original or scan/import modules. registrations provides
inspectable discovery of supplied adapters. Duplicate reasoner IDs are rejected.
Family and version are required strings; future/custom families need no core
changes. Callables stay in the registry, never in serialized context/results.
Metadata includes requirements, priority, enabled status, and adapter version.

Execution order is ascending integer priority, then reasoner_id, independent of
registration order. Execution order is not truth priority. There is no dependency
graph, inter-reasoner output feed, adaptive routing, voting, or consensus. All
eligible reasoners receive the same immutable input context. Callables must be
deterministic and bounded; the orchestrator cannot make arbitrary user code pure.
Version identifiers must change when adapter behavior changes: execution IDs
fingerprint context and registration metadata, not callable bytecode or closure
state. No wall-clock timestamps, random IDs, or timing metrics enter results.

## Eligibility

Every registration yields an Eligibility record before execution. Statuses:
eligible, missing_required_evidence, incompatible_context, unsupported_input,
disabled. Required evidence families and evidence_type categories must have
actual non-unavailable items; an empty-but-present feed is insufficient.
required_statuses can require, for example, sensory observed instead of merely
sensory available. Optional evidence is descriptive and never blocks execution.
Required capabilities are caller-declared strings, not raw capability objects.
Schema and optional coordinate-frame requirements are checked explicitly.

Family/category requirements are independent inventory checks, not semantic
inference about item content. Missing geometry inside a physical summary is not
automatically resolved by eligibility. Adapters must validate any additional
domain-specific input they require. A skipped registration emits no output or
negative evidence, and missing evidence does not falsify a hypothesis.

## Context, results, and epistemic boundaries

OrchestrationContext holds the existing frozen common state, detached bounded
configuration, capability names, and a deterministic context reference. Session,
scene, and timestamp are read from the common state, not separately overridable.
Declared configuration/output context mismatches and source-time leakage are
rejected. Outputs cannot use future source timestamps. Prediction target times
remain distinct; hypotheses must satisfy target = created time + horizon.

ReasonerOutput is a transport envelope: structured_output, existing EvidenceItem
candidates, existing Hypothesis objects, uncertainty, and notes. The helper
from_expected_consequences freezes predictions and labels their role expected.
There is no automatic WorldState reconstruction from common evidence, no hidden
actor mapping, and no latent-to-physical interpretation.

ReasonerResult carries executed/skipped/failed, eligibility, producer identity,
registration metadata, context reference, session/scene/time, and exposed input
evidence IDs. These IDs describe the accessible inventory, not causal use or
support. Candidates/hypotheses must explicitly declare producer_reasoner_id,
session_id, scene_id, timestamp, and derived_from input IDs in provenance.
Candidate evidence requires an explicit epistemic status. Hypothesis support/
opposition IDs must reference inputs. Their original statuses, confidence and
probability fields are retained; no normalization or belief update is invoked.

collision_possible remains possibility; expected_sound remains expected;
learned_signal remains internal signal. Explicit source classifications are
preserved, not scientifically verified by the executor. An output envelope is
never an observation or truth endorsement. Outputs are not inserted into or
merged with CommonEvidenceState. Multiple agreeing results remain separate.

## Isolation and limits

Exceptions from execution, unsupported return types, or invalid outputs produce
a failed result while independent reasoners continue. Error metadata contains a
type name truncated to 80 characters and a fixed code. Arbitrary exception
messages, paths, secrets, reprs, and tracebacks are not serialized. Output
validation failures discard that reasoner's output. KeyboardInterrupt and other
BaseException process-control signals are deliberately not swallowed.

Frozen contexts, registrations, registries, outputs, and result metadata prevent
ordinary caller mutation. Existing bounded_plain/freeze conventions enforce
finite, detached structured metadata and reject raw tensor/embedding/model-style
payloads. The Step 14 node/depth/string and 1 MiB serialization limits apply;
large result inventories should be split by the caller. An aggregate bound
failure can reject the final result instead of silently truncating it.

This is in-process exception isolation, not a security sandbox or worker-process
timeout system. A hostile or hanging callable, external I/O, closure mutation,
and process-global side effects are outside that guarantee. No threads, workers,
GPU objects, models, downloads, or production-global mutations are introduced.

## Future layers

Step 14 CommonEvidenceState -> Step 15 deterministic orchestration -> Step 16
future Dynamic Reasoning Connectome integration. A future routing component may
supply explicit registrations/selections to this executor; none is implemented
here. No learned weights, graph plasticity, Bayesian logic, or ASTRA-EFA exists
in this module.

Future scientific families may register callables, but no MethodReasoner,
FigureEvidenceReasoner, or parser is implemented. Synthetic tests preserve
reported_result != author_claim != aisthesis_inference. They demonstrate transport
compatibility, not scientific reasoning or experimental usefulness.

## Validation

New tests first:

```powershell
.\.venv\Scripts\python.exe -m pytest evaluation/tests/test_reasoner_orchestration.py -q
```

48 tests passed. Related common/evidence/world-model/physics/hypothesis/sensory
and safe temporal/occlusion regressions: 397 passed, 108 subtests passed.

The broader safe selection follows HYBRID_WORLD_MODEL_V01.md: evaluation/tests
plus root files defining test functions or test methods. The same 13 script-style
files are excluded because known import-time model/perception calls make an
unrestricted pytest run unsuitable for this no-model-execution task. The existing
untracked test_prediction_experience.py is included read-only. No unrestricted
full-suite success is claimed. No commit, push, tag, or generated-file staging.

Broader safe regression: **706 passed, 130 subtests passed** (658 existing tests
plus 48 new tests).
