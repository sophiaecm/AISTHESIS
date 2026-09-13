# Dynamic Reasoning Connectome v0.1

Status: **ENGINEERING MILESTONE**.

- IMPLEMENTED
- TESTED (synthetic engineering correctness)
- NOT YET TASK-LEVEL EVALUATED
- NOT YET EXPERIMENTALLY SUPPORTED

Relevance is not truth probability.

Routing priority is not epistemic confidence.

Selection does not imply correctness.

Non-selection does not imply falsity.

The Dynamic Reasoning Connectome v0.1 is a deterministic routing
abstraction, not a biological connectome model.

The Dynamic Reasoning Connectome is an engineering routing abstraction,
not a biological connectome model.

The graph routes reasoning; it does not represent the physical world.

## Repository fit and logical flow

Step 14 CommonEvidenceState -> Step 16 Dynamic Reasoning Connectome ->
Step 15 ReasonerOrchestrator -> ReasonerResult.

This ordering is logical routing flow; roadmap numbering does not imply runtime
sequencing constraints. Existing perception/scene_graph.py represents detected
objects and spatial relations, not routing capabilities. Reusing that ontology
would conflate routing edges with physical relations. No preexisting relevance
router was found. A single additive fifth_layer/reasoning_connectome.py module
reuses Step 15 registration, context, eligibility and execution without changing
them. Existing engines, reasoners, CommonEvidenceState, and tests are unchanged.

## API and manual integration

```python
from fifth_layer.reasoning_connectome import DynamicReasoningConnectome, ConnectomeConfig

# registry is an existing Step 15 ReasonerRegistry.
# state may be CommonEvidenceState or OrchestrationContext.
router = DynamicReasoningConnectome(ConnectomeConfig(
    min_relevance=0.3, max_selected_reasoners=3,
))
snapshot = router.route(state, registry)       # Does not execute reasoners.
print(snapshot.to_json())
result = router.execute(state, registry, snapshot)  # Explicit opt-in bridge.
```

ConnectomeContext.from_state extracts immutable descriptors and may also be
passed to route/execute. It retains the Step 15 context; no second source of
session, scene or time is introduced. A node uses the existing reasoner_id as
node_id and references registration metadata: family, version, requirements,
optional sources, capabilities, original priority, schema/frame restrictions.
No callable or model is stored in the graph. The Step 15 registry owns identity
validation and rejects duplicate reasoner IDs; there is no second node registry.

## Descriptive context and explicit scoring

Descriptors copy evidence ID, source family, evidence_type category, epistemic
status and source time. Source references retain the original per-item provenance.
The context reference fingerprints the full Step 14 input and configuration;
the snapshot does not embed physical payloads or raw learned representations.
Source uncertainty is preserved. Empty inventory is marked missing_context;
incomplete integration context or unknown/unavailable statuses mark partial_context.
No uncertainty flag is used to reduce a score. Unknown does not mean irrelevant.

Default score is the clipped sum of four explicit nonnegative components:

| Contribution | Default | Trigger |
| --- | ---: | --- |
| base_relevance | 0.2 | Every registered node, including empty/unknown context |
| evidence_match | 0.4 | At least one non-unavailable matching source family |
| category_match | 0.2 | At least one matching required_evidence_types category |
| capability_match | 0.1 | At least one declared required capability is available |

Each contribution is applied at most once. Matching evidence IDs and capability
names are recorded. Multiple sources or repeated statements do not accumulate
extra weight. Default family matches are physics -> physical/physics/
physics_constraint, physical -> physical, temporal -> temporal, occlusion ->
occlusion, sensory -> sensory, learned_representation -> learned_representation,
and documentary -> documentary. Registered required/optional source families
also participate in family matching. Custom reasoner-family mappings can be
supplied through ConnectomeConfig.family_evidence.

Only explicit inventory fields are inspected: temporal evidence presence does
not establish temporal continuity; physical evidence does not establish collision;
expected sensory evidence does not establish an observation. There is no semantic
payload parsing, inferred object dynamics, or learned-to-physical interpretation.
Category requirements are matched as literal strings. Eligibility remains the
authority for required statuses, capabilities and source requirements.

Scores and increments are bounded to [0, 1]. The sum is clipped to 1, with
unclipped_relevance and clipping_adjustment recorded in the breakdown. These are
human-configured routing heuristics, not scientifically calibrated values.
0.8 relevance does not mean an 80% probability of correctness. Source confidence,
probability, agreement, successes, failures, rewards and experience values are
not scorer inputs. Experience-family records are excluded from family/category
bonuses; config cannot map a reasoner family to experience. They may remain
inspectable in the original input and source references, without adapting weights.

## Selection and Step 15 safeguards

Selection sorts by descending relevance then reasoner_id. It requires BOTH an
eligible Step 15 plan entry and score >= min_relevance, then applies the optional
max_selected_reasoners limit. Zero limit explicitly selects none. Original
registration priority is retained in graph metadata but does not break routing
score ties. Exclusions identify step15_ineligible (with original eligibility
status/reasons), below_relevance_threshold, or selection_limit.

High relevance cannot enable a disabled, missing-evidence, unsupported or
frame-incompatible reasoner. Unavailable or empty-but-present feeds do not count
as matching evidence. Unknown source status remains unknown; matching merely
means its source family is relevant to inspect. Non-selection emits no negative
evidence, and selection emits no support claim.

The explicit execute bridge recalculates routing against the supplied context,
registry metadata and config. A mismatched, stale or modified snapshot is rejected
before any callable runs. This binds selection to session/scene/time as well as
evidence content and policy. Upstream Step 14/15 validation rejects declared
future source data and incompatible input context.

Only selected registrations are copied into a temporary Step 15 registry, with
integer priorities assigned from the relevance ordering. All other registration
fields and callables remain unchanged. The original registry is not mutated.
ReasonerOrchestrator then rechecks eligibility and executes normally. Its failure
semantics, evidence/hypothesis statuses and confidence values are unchanged.
Excluded nodes appear in the graph, not as manufactured execution outputs.

## Graph and immutable snapshot

Edges have only context_activates_reasoner semantics. The source endpoint is the
snapshot's context_reference, and the target is an existing reasoner ID. Each
positive matched rule produces an edge with its contribution and source evidence
or capability references. Base relevance is a policy constant, so it creates no
context activation edge. Edges may describe relevance to an ineligible node;
they do not grant execution permission.

No reasoner-to-reasoner dependencies or causal edges are inferred. Reusing the
same evidence reference on two edges does not duplicate evidence or make it
independent. The graph is routing metadata, never another world state.

ReasoningGraphSnapshot contains context identity, nodes, edges, bounded scores,
contribution breakdowns, selected/excluded IDs and reasons, config, source
references, descriptive context and uncertainty. Maps and sequences are detached
and frozen; JSON serialization has deterministic key and node/rule ordering.
Step 14 bounded_plain/freeze restrictions apply, including 50,000 traversal nodes,
depth 24 and 1 MiB per serialization boundary. Large inventories fail explicitly;
raw arrays/tensors/models and embedding payloads are not accepted.

## Determinism, limits and future compatibility

There is no random routing, clock input, training, online adaptation, persistent
weight update, experience feedback, ASTRA-EFA import, V-JEPA import, neural model,
parser, Bayesian belief or truth fusion. Prior snapshots/results never affect a
later route call. Changing only an evidence family can change relevance, according
to the same static rule. No production connection is installed.

Scoring, selection, immutable graph representation and the execution bridge are
separate functions/contracts so a later Step 23 policy can replace scoring.
No adaptive v0.2 policy or experience-history interface is implemented. Custom
scientific families may be registered/mapped, and documentary categories/statuses
remain distinct; no scientific reasoner or Mini-Lab parser is implemented here.

The bridge fingerprints registration metadata, not callable bytecode/closure
state, matching Step 15's versioning convention. Adapter authors must update
versions when behavior changes. Execution remains in-process and inherits Step
15's callable purity, timeout and isolation limits. Routing itself never invokes
the registered callables. Snapshot comparison is an integrity check, not a
cryptographic authorization boundary for hostile Python code.

## Engineering validation

New tests first:

```powershell
.\.venv\Scripts\python.exe -m pytest evaluation/tests/test_reasoning_connectome.py -q
```

54 new tests passed. Related Step 15, CommonEvidenceState, world-model/evidence,
physics/hypothesis/sensory and safe temporal/occlusion selection: 445 passed,
108 subtests passed. Step 15 tests were not modified.

The broader safe regression selection follows HYBRID_WORLD_MODEL_V01.md:
evaluation/tests plus root files defining test functions/methods. The same 13
script-style files with known import-time model/perception side effects are
excluded; unrestricted pytest is not rerun or claimed successful. Existing
untracked test_prediction_experience.py is included read-only. No commit, push,
tag, generated-file staging or production behavior changes are part of this work.

Broader safe regression: **760 passed, 130 subtests passed** (706 existing tests
plus 54 new tests).
