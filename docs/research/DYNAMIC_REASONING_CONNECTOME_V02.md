# Step 23 — Dynamic Reasoning Connectome v0.2

## 1. Purpose

Route registered reasoners using explicit multi-layer context descriptors. The
router produces a separate immutable graph; it performs no inference, truth
fusion, belief update, branch selection or physical action. Relevance is a
caller-configured engineering heuristic.

## 2. Relationship to Dynamic Reasoning Connectome v0.1

The original `reasoning_connectome.py` is unchanged. v0.2 lives in a separate
module with independent profiles, configuration, context and snapshot contracts.
It does not infer preferences from reasoner IDs, class names or families.
Unprofiled reasoners receive **base relevance only**, even with rich evidence;
Step 15 eligibility still applies. This is intentionally not the v0.1 family
bonus fallback. Existing experience EvidenceItems are never routing bonus sources.

## 3. Relationship to Reasoner Orchestration v0.2

The existing OrchestrationContext, ReasonerRegistration, ReasonerRegistry and
ReasonerOrchestrator are reused unchanged. `plan` is authoritative for execution
eligibility. `execute` delegates only to the existing orchestrator, preserving
output validation, exception containment and eligibility rechecks.

```python
from fifth_layer.orchestration import OrchestrationContext
from fifth_layer.reasoning_connectome_v02 import (
    MultiLayerReasoningContext, ReasonerRoutingProfile,
    DynamicReasoningConnectomeV02,
)

context = MultiLayerReasoningContext(
    OrchestrationContext(common_evidence_state),
    physical_state=physical_state,
    belief_state=belief_state,
)
profiles = (ReasonerRoutingProfile(
    reasoner_id="registered_reasoner",
    preferred_layers=("physical", "belief"),
    preferred_descriptor_categories=("physical.motion_state_present",),
    provenance={"source": "explicit caller configuration"},
),)
router = DynamicReasoningConnectomeV02()
snapshot = router.route(context, registry, profiles)
# Explicit optional execution, never performed by route():
result = router.execute(context, registry, profiles, snapshot)
```

## 4. Multi-layer context

`MultiLayerReasoningContext` requires OrchestrationContext containing
CommonEvidenceState v0.1. Optional fields and accepted contracts are:

| Field / routing layer | Accepted source |
| --- | --- |
| `physical_state` / `physical` | LatentPhysicalState, latent-physical-state-0.1 |
| `physics_constraints` / `physics_constraints` | PhysicsConstraintBundle, physics-constraints-0.2; or PhysicsTransitionAssessment, physics-transition-0.2 |
| `cross_modal_consequences` / `cross_modal` | CrossModalConsequenceBundle, cross-modal-consequences-0.2 |
| `multiple_futures` / `multiple_futures` | MultipleFutureBundle, multiple-futures-0.2 |
| `belief_state` / `belief` | BayesianBeliefState, bayesian-belief-state-v0.1 |
| `experience_view` / `experience` | ExperienceLearningViewV02, experience-learning-v0.2 |

The mandatory common evidence layer is named `evidence`. Every optional layer
can be supplied alone. Immutable source references are retained in memory;
serialization includes content references, descriptors, availability and
uncertainty instead of repeatedly embedding full source trees. The context has
derived `context_id`, `descriptors`, `layer_availability`, `source_references`,
`uncertainty`, and schema `reasoning-connectome-v0.2`.

## 5. Context alignment

Supplied current layers must exactly match common scene, session, timestamp and
coordinate frame wherever the source contract exposes those fields. Unknown
timestamps match only unknown timestamps; known earlier snapshots are not silently
treated as current. Declared nested source times cannot exceed current time, and
nested declared sessions/frames must agree. Existing `_alignment` is reused for
nonhistorical layers; retrospective transition history may precede current time.
Constraint bundles lack top-level session/frame: declared nested metadata is
checked, but missing fields are not fabricated.

When multiple layers are supplied, physical source identities must agree, future
consequence/constraint references must exist in supplied bundles, and belief
source bundle identity and complete branch records must match supplied futures.

Experience VIEW headers and current query headers/frame/provenance must align.
Queries, rows and retrieval references are checked. Historical record timestamps
must be strictly past and their stored prediction session must match. Historical
scenes and frames are not reinterpreted as current scenes/frames. If a belief is
supplied, its ID, hypothesis inventory and base posteriors must match the view.
Without that optional belief, its reference remains attributed provenance rather
than a resolved source. Misalignment raises ValueError; nothing is reconciled.

## 6. Routing descriptors

Each immutable descriptor has `descriptor_id`, `layer`, `category`, `source_refs`,
`routable` and `details`. Category names are exact dot-separated strings:

| Layer | Categories |
| --- | --- |
| Evidence | `evidence.family.<source>`, `evidence.type.<type>`, `evidence.status.<status-or-unspecified>` |
| Physical | `physical.present`, `physical.motion_state_present`, `physical.contact_candidate_present`, `physical.collision_risk_geometry_present`, `physical.tracking_or_motion_discontinuity_present` |
| Constraints | `physics_constraints.present`, `.type.<constraint_type>`, `.status.<status>` |
| Cross-modal | `cross_modal.present`, `.modality.<modality>`, `.status.<status>`, `.event_family.<event_family>` |
| Futures | `multiple_futures.present`, `.family.<branch_family>`, `.status.<status>`, `.horizon.<kind>`, `.relation.<relation>` |
| Belief | `belief.present`, `.initialization.<mode>`, `.update.<status>`, `.posterior.available` or `.posterior.unavailable` |
| Experience | `experience.present`, `.enabled`, `.disabled`, `.relevant_history_present`, `.positive_context_present`, `.negative_context_present`, `.neutral_only`, `.suppressed_rows_present`, `.contribution_available` |

Every absent optional layer has `<layer>.unavailable`, marked non-routable.
Physical, cross-modal, futures, belief and enabled experience have non-routable
`<layer>.inventory` descriptors with counts. Counts are audit data only.
Evidence details preserve evidence ID, family, type, status and timestamp, never
confidence or value magnitude. Source IDs are references, never semantic weights.

## 7. ReasonerRoutingProfile

Frozen fields: `reasoner_id`, `preferred_layers`,
`preferred_descriptor_categories`, `optional_descriptor_categories`,
`blocked_when_flags`, `routing_role`, `provenance`.
Profiles are explicitly caller supplied, unique per registered reasoner. Unknown
reasoner IDs and layer names are rejected. String sequences are canonicalized.
Unknown category strings simply do not match; no semantic guessing occurs.

Preferred layers each enable one presence bonus. Preferred and optional
descriptor categories share **one** bonus pool; neither imposes an eligibility
requirement. `routing_role` and provenance are descriptive only.
`blocked_when_flags` matches exact context uncertainty flags and can exclude a
reasoner. Unprofiled reasoners receive no bonuses or inferred preferences.

## 8. Relevance heuristic

`ConnectomeConfigV02` defaults:

| Setting | Default |
| --- | ---: |
| min_relevance | 0.30 |
| max_selected_reasoners | None (no additional selection limit) |
| base_relevance | 0.20 |
| evidence_layer_increment | 0.20 |
| physical_layer_increment | 0.10 |
| physics_constraint_increment | 0.10 |
| cross_modal_increment | 0.10 |
| future_layer_increment | 0.10 |
| belief_layer_increment | 0.10 |
| experience_layer_increment | 0.05 |
| descriptor_match_increment | 0.10 |

All numeric settings above, except the count limit, must be finite nonboolean
numbers in [0,1]. The limit is None or a nonnegative integer; zero selects none.

```text
raw = base + sum(one increment per preferred present routable layer)
           + (descriptor increment if any configured routable category matches)
relevance = clip(raw, 0, 1)
```

`math.fsum` is used. No count multiplication, normalization, softmax, probability
magnitude, recency weighting or uncertainty penalty is applied. A supplied empty
optional layer can supply its `present` bonus; presence describes an interface,
not evidence sufficiency. An empty evidence feed supplies no evidence bonus.

## 9. Why relevance is not probability

The score is a configurable routing preference. It is not truth probability,
confidence, posterior, scientific support or proof that a selected reasoner is
correct. Multiple scores may each equal one and need not sum to one.

## 10. Eligibility dominance

`ReasonerOrchestrator.plan` is called on the original OrchestrationContext.
Disabled, unsupported_input, incompatible_context and missing_required_evidence
always exclude a reasoner, regardless of score or profile. New optional layers
are never inserted into CommonEvidenceState or used to satisfy required evidence.
Required observed sensory statuses remain the orchestrator's responsibility.

## 11. Physical layer routing

Motion presence requires an explicit non-null motion attribute with observed or
estimated status. Contact, collision geometry and discontinuity presence require
the exact existing relation/dynamics signal name. Source statuses are preserved.
Object count and active constraint count are annotations only. No tracker-ID
comparison, hidden actor inference or new physical predicate is computed.

## 12. Physics-constraint routing

Preserve exact constraint type and satisfied/violated/indeterminate/unsupported/
not_applicable category. Per-result uncertainties are retained by constraint ID.
A violation is an assessment category, not confirmation of event occurrence.
The transition assessment's end context must be current; its prior time remains
retrospective provenance.

## 13. Cross-modal routing

Preserve candidate modality, event family and status, including expected,
possible, indeterminate, unavailable and unsupported. Candidate descriptors state
`consequence_candidate_not_observation`. Repeated candidates never multiply
bonuses. A candidate sound is not heard sound, and cannot satisfy an observed
sensory input requirement.

## 14. Multiple-futures routing

Preserve branch families, statuses, horizons and relation types. References and
counts remain audit metadata. No branch is selected, collapsed or promoted to
fact. Mutually exclusive relations describe branch assumptions, not which event
occurred. More branches do not increase a layer/category bonus.

## 15. Bayesian-belief routing

Use only initialization/update categories, posterior availability and layer
presence. Hypothesis count is non-routable audit information. No highest-posterior
identity or numeric probability affects relevance. Changing probabilities can
change context/source references and therefore invalidate a prior snapshot while
leaving routing scores and selection unchanged. No likelihood is generated and
no Bayesian update is called.

## 16. Experience-context routing

Disabled experience has availability `disabled` and no routable descriptors or
bonus. Enabled experience exposes categorical retrieval presence, positive or
negative contribution sign, all-zero contribution context with retrieval
(`neutral_only`), suppressed-row presence and base-posterior availability.
Neutral-only includes cancellation or suppression; it is not a claim that all
retrieved statuses were neutral. Retrieval count is annotation only.

Exact historical contribution and experience-informed score magnitudes never
enter relevance. No learning call, mutation, evidence admission or historical
observation promotion occurs. Source view references may change when amounts or
history change even if relevance remains identical.

## 17. Missing/unknown semantics

Absent layers are unavailable, not false. Unknown statuses stay categories, not
negative evidence. An unavailable belief distribution may still be a present
belief-layer context. A missing layer is not automatically penalized. Experience
EvidenceItems already in common evidence are exposed as non-routable descriptors,
preserving v0.1's exclusion of that family from routing bonuses.

## 18. Uncertainty

Source uncertainty is retained by layer. Optional-layer absence, disabled
experience and indeterminate/unavailable/unsupported category flags are explicit.
String uncertainties are prefixed by layer for exact profile blocking. Common
integration flags are retained. No `1-confidence` conversion or pooled uncertainty
metric exists. Uncertainty has zero score penalty; only an explicit profile block
can exclude on a flag.

## 19. Selection

Compute existing eligibility and v0.2 relevance; sort by relevance descending,
then reasoner ID. Exclude Step 15 ineligible, profile-blocked and below-threshold
reasoners, then apply the optional selection limit. Each exclusion carries its
reason, eligibility and blocked flags. No winner, truth or action is selected.

## 20. Routing graph

`ReasoningGraphSnapshotV02` records session/scene/time, context/registry/profile
references, nodes, edges, scores, score breakdown, selected IDs, exclusion reasons,
availability, source references, uncertainty, descriptors, config, provenance,
schema and stable snapshot ID. Nodes are exactly registered reasoners.

Edges use `context_layer_activates_reasoner`, with source layer, context reference,
target reasoner, routing rule, increment, descriptor references/categories and
`truth_semantics=False`. Layer-presence edges contribute once per preferred layer.
The single descriptor bonus is attributed to the first canonical matching
descriptor and its source layer; the breakdown retains all matched categories.
No causal or biological reasoner-to-reasoner edges are invented.

## 21. Execution bridge

`execute(context, registry, profiles, snapshot)` reruns routing and requires exact
canonical snapshot equality. Context, profiles, registry metadata, config or
snapshot changes invalidate execution. Sources are revalidated during routing.

Frozen registrations intentionally omit callable bodies from metadata. v0.2
therefore also retains private in-process `(reasoner_id, callable)` bindings in
the snapshot and requires identical callable objects for execution. These are
excluded from JSON/content IDs, so serialization remains deterministic. A changed
callback requires rerouting even with identical registration metadata. Snapshot
JSON alone is an audit artifact, not a portable executable authorization.
Mutation inside a callback's closure is outside the frozen registration contract;
callers must manage their implementations/version metadata.

Selected registrations are reused with priority copied to selection order in a
new registry. The existing orchestrator rechecks eligibility and executes them.
Unselected reasoners are not called. Failed execution retains existing containment.
Reasoners receive the original OrchestrationContext, not injected optional-layer
data; this step routes execution without changing reasoner input contracts.
No output feeds back into context, and no recursive routing occurs.

## 22. Determinism

Stable IDs use canonical structured content; profiles, nodes, descriptors, edges,
breakdowns, flags and selections are ordered deterministically. Numeric magnitudes
can affect source identity but never routing weights. No randomness, wall-clock,
unstable repr or callback memory address enters serialized IDs.

Existing bounded serialization limits apply. Source validation and hashing can
be expensive for large historical views; no performance improvement is claimed.
No new caching/persistence system is introduced.

## 23. Explicit non-goals

No inference, truth fusion, Bayesian update, likelihood generation, branch/MAP
selection, experience learning, observation promotion, hidden actor inference,
physical action, application wiring, calibration or scientific validation.
No dependencies, models, repositories, datasets, GPU or internet access are added.
Exactly three Step 23 files are created; frozen production, live_app and Mini-Lab
files are untouched. No staging, committing, pushing or tagging is performed.

## 24. Relationship to Step 24 Complex Systems Layer

The explicit layer/category/profile pattern can later host an optional complex
systems source with its own contract, alignment and descriptor extractor. No
interaction/stability/emergence inference or complex-systems implementation is
included now. New descriptors require an explicit versioned integration, not
automatic layer discovery.

## 25. Relationship to Experiment 001

No improved reasoning, prediction, efficiency, TTA, hidden actor anticipation,
calibration, cross-modal benefit, experience benefit or Experiment 001 success is
established. Later controlled evaluation can compare all/static reasoners, v0.1
evidence-only routing and v0.2 multi-layer routing under fixed inputs/task metrics.

## 26. Validation status

- 🧩 IMPLEMENTED
- 🧪 TESTED (contract and regression verification only)
- 📊 NOT YET TASK-LEVEL EVALUATED
- 🏆 NOT YET EXPERIMENTALLY SUPPORTED

Final combined run on 2026-09-14: **1,016 passed in 43.67 seconds**.

| Suite | Passed |
| --- | ---: |
| Step 23 dedicated | 87 |
| Existing connectome v0.1 | 54 |
| Existing orchestration | 48 |
| Step 22 experience learning | 80 |
| Step 21 trajectory | 62 |
| Step 20 bridge | 103 |
| Step 19 beliefs | 117 |
| Step 18 futures | 144 |
| Step 17 consequences | 93 |
| Physical world, constraints, latent, hybrid, common evidence | 228 |

```powershell
.\.venv\Scripts\python.exe -m pytest evaluation/tests/test_reasoning_connectome_v02.py evaluation/tests/test_reasoning_connectome.py evaluation/tests/test_reasoner_orchestration.py evaluation/tests/test_experience_learning_v02.py evaluation/tests/test_astra_efa_experience.py evaluation/tests/test_belief_prediction_bridge.py evaluation/tests/test_bayesian_belief_state.py evaluation/tests/test_multiple_futures.py evaluation/tests/test_cross_modal_consequences.py evaluation/tests/test_physical_world_model.py evaluation/tests/test_physics_constraints.py evaluation/tests/test_latent_physical_state.py evaluation/tests/test_hybrid_world_state.py evaluation/tests/test_common_evidence_state.py -q
```

Tests cover aligned optional/complete contexts, malformed input, source linkage,
profiles, descriptors, categorical scoring, count and magnitude independence,
eligibility dominance, deterministic selection, immutable graphs, stale/tampered
snapshots and callback replacement, and execution through the existing
orchestrator. No existing test was changed or weakened.

## 27. Canonical epistemic statements

> The connectome routes reasoning; it does not determine truth.

> Routing relevance is an engineering heuristic, not a probability.

> A selected reasoner is not necessarily the correct reasoner.

> An excluded reasoner is not a false reasoner.

> Step 15 eligibility cannot be overridden by Step 23 relevance.

> A missing layer is unavailable, not negative evidence.

> Physics plausibility is not event occurrence.

> A cross-modal consequence candidate is not a sensory observation.

> A possible future is not a predicted fact.

> A Bayesian posterior is not truth.

> Experience-informed context is historical context, not current observation.

> Posterior magnitude is not used as routing confidence in v0.2.

> Historical contribution magnitude is not used as routing confidence in v0.2.

> Step 23 does not select a future branch.

> Step 23 does not update Bayesian beliefs.

> Step 23 does not learn from experience.

> Step 23 does not infer hidden actors.

> Step 23 does not execute physical actions.

> Step 23 is an engineering routing layer, not scientific validation.
