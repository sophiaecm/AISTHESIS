# Step 24 — Complex Systems Layer v0.1

## 1. Purpose

An optional, immutable systems-structure representation grounded in existing
physical signals. It describes interaction edges, connected subsystem candidates,
categorical stability indicators and conditional transition candidates. It does
not establish causal mechanisms, emergence, complexity benefit or scientific validity.

## 2. Why systems-level representation is needed

Explicit relations and constraint scopes can connect multiple physical objects.
The layer preserves that structure and its uncertainty separately from isolated
object attributes. Coexistence, class similarity and proximity alone create no
edge. Missing coupling does not establish independence.

## 3. Inputs and API

```python
from fifth_layer.world_model.complex_systems import ComplexSystemsBuilder

state = ComplexSystemsBuilder().build(
    physical_state,
    physics_constraints=None,
    cross_modal_consequences=None,
    multiple_futures=None,
)
```

LatentPhysicalState `latent-physical-state-0.1` is mandatory. Optional inputs are
PhysicsConstraintBundle `physics-constraints-0.2` or PhysicsTransitionAssessment
`physics-transition-0.2`, CrossModalConsequenceBundle `cross-modal-consequences-0.2`,
and MultipleFutureBundle `multiple-futures-0.2`. No belief, experience or router
input is required. All new records have schema `complex-systems-v0.1`.

## 4. Interaction graph

Exact implemented source-to-type mapping:

| Interaction type | Source |
| --- | --- |
| `contact_relation_candidate` | Latent relation `signal=contact_candidate` |
| `collision_relation_candidate` | Latent relation `signal=collision_risk_geometry` |
| `support_relation_candidate` | Latent relation `signal=support_candidate` |
| `discontinuity_relation_candidate` | Latent dynamics `signal=tracking_or_motion_discontinuity`, with at least two objects |
| `shared_constraint_relation` | Explicit active/current constraint object scope containing at least two objects |

Relation values explicitly false or missing/None are not admitted. Every edge
requires source references and at least two distinct current objects. No spatial
test, class-based relation, motion-coupling inference or causal direction is added.
Constraints supply a shared scope, not proof of mechanical interaction. Cross-modal
and future sources enrich exact-object-scope edges only; they never create edges.

## 5. InteractionEdge semantics

Fields: `interaction_id`, aligned `scene_id`, `timestamp`, `session_id`,
`coordinate_frame_id`, `object_ids`, `interaction_type`, `status`,
`source_constraint_ids`, `source_consequence_ids`, `source_future_ids`,
`source_field_references`, `uncertainty`, `provenance`, `schema_version`.

Source observed/estimated/satisfied maps to `present_candidate`; possible stays
`possible`; unknown/indeterminate/violated maps to `indeterminate`; unsupported
and unavailable stay as such; not_applicable maps to `unavailable`. Other status
labels are skipped with an extraction audit. None of these means confirmed cause.

Edges are undirected. Original endpoint orientation, source status, fields and
references remain in provenance. Sources with the same type, sorted object set
and mapped status are merged with unique source records and reference unions.
Different statuses remain separate auditable edges. Cross-modal/future enrichment
preserves full source records without treating modality or branch count as strength.

## 6. CoupledSubsystemCandidate

Fields: `subsystem_id`, aligned context, `object_ids`, `interaction_ids`,
`coupling_basis`, `status`, `uncertainty`, `provenance`, `schema_version`.

Deterministic standard-library connected components use only present_candidate,
possible and indeterminate edges. Multi-object constraint scopes act as undirected
group links. Unsupported/unavailable edges remain audit records but do not connect
objects. Components require at least two objects; singletons are never called
coupled subsystems. Each component retains all participating interaction IDs.
Its status is indeterminate if any participating edge is indeterminate, otherwise
possible. Connectivity is a candidate grouping, not physical causal coupling.

## 7. StabilityAssessmentCandidate

Fields: `assessment_id`, `subsystem_id`, `scene_id`, `timestamp`, `status`,
`indicators`, `counter_indicators`, `source_refs`, `uncertainty`, `provenance`,
`schema_version`. One assessment is generated per subsystem.

Positive indicators use exact constraint triples:

- support_stability_possibility / support_geometry_possible / satisfied
- inertia_consistency / motion_continuity_consistent / satisfied

Negative indicators use exact constraint triples:

- inertia_consistency / abrupt_change_detected / violated
- implausible_displacement / kinematic_outlier / violated

Explicit possible/present collision or discontinuity interaction edges also supply
negative indicators. Possible/estimated latent discontinuity dynamics with objects
inside a subsystem supply a negative indicator even when the dynamic signal itself
is a singleton. Ordinary motion and generic violations are not negative indicators.

Rules, in order:

1. Positive and negative indicators: `mixed`.
2. Negative indicators only: `unstable_candidate`.
3. Positive indicators covering every subsystem object, with no indeterminate
   participating edge or relevant constraint: `stable_candidate`.
4. Otherwise: `indeterminate`.

Stable candidate means explicit geometry/sampled-continuity support under this
rule, not mechanically proven stability. Partial positive coverage is insufficient.
Absence of violation, uncertainty, stationary labels or prediction errors create
no indicator. A support candidate relation alone does not establish stability.
The frozen engine does not expose a verified support-loss/change contract; no
generic support-violation or catastrophic failure inference is fabricated.

## 8. TransitionPatternCandidate

Fields: `transition_id`, `subsystem_id`, `pattern_type`, `status`, `source_refs`,
`assumptions`, `uncertainty`, `provenance`, `schema_version`.

Only supplied Step 18 branches with objects contained in a subsystem are considered:

- Contact-resolution branch plus an exact-object-scope contact/collision edge:
  `contact_transition_candidate`, retaining contact and no-contact alternatives.
- The same source with `contact_within_horizon` and collision edge:
  additional `collision_transition_candidate`.
- Discontinuity-resolution branch plus an exact-object-scope explicit latent
  discontinuity dynamic: `continuity_transition_candidate`.
- A grounded branch with `outcome_unresolved`: `unresolved_transition` instead
  of the generic contact/continuity label.

Branch possible/indeterminate/unsupported/unavailable status is preserved;
unresolved usable branches remain indeterminate. Each transition retains the full
branch and assumptions and explicitly sets `occurred=False`. No winner is chosen.
No support_change_candidate, cascade or propagation rule is implemented.

## 9. Uncertainty

Sorted categorical unions preserve physical, object, relation, dynamics, constraint,
consequence and future uncertainty. The conservative union is propagated across
generated records; it is not a relevance-weighted or numerical uncertainty estimate.
Additional categories include relation_direction_unknown,
subsystem_causality_unknown, missing_interaction_not_independence,
constraint_indeterminate, interaction_indeterminate, cross_modal_candidate_only,
future_branch_non_exhaustive and temporal_alignment_unknown as applicable.

Missing optional layers are recorded as `unavailable` in provenance and named
uncertainty flags. They never supply a negative indicator. Source statuses and
full supplied snapshots remain provenance for inspection.

## 10. Alignment

All exposed current scene/session/timestamp/frame fields must match the physical
state exactly. Existing `_alignment` rejects future/unorderable nested source
times and conflicting declared sessions/frames. Constraint bundles lacking
top-level session/frame use their declared source metadata; absent declarations
are not invented. Retrospective assessment history remains earlier provenance.

Declared physical-state IDs must agree. Optional branch/candidate and constraint
object IDs must exist in the current physical state. This v0.1 current-object
scope rejects constraints referencing historical-only objects rather than guessing
identity continuity. Referenced constraints must resolve through active summaries
or the supplied constraint bundle. Matching IDs with inconsistent type/status/
finding/object scope are rejected. If a consequence bundle is supplied, branch
consequence IDs must exist in it. Without that optional bundle, branch consequence
references remain unresolved provenance only. Inputs are never silently repaired.

## 11. Determinism

All generated IDs use `stable_id` over canonical data. Object/reference sets,
source unions, graph traversal, component construction, output records and audit
entries are deterministically ordered. Duplicate physical inventories and duplicate
semantic sources are canonicalized; source provenance is retained. Existing IDs
are treated as source references, not authenticated content identities.
No randomness, wall-clock, unstable repr, dependency or model is used.

Records use frozen dataclasses, tuples and detached read-only mappings. Serialization
uses sorted compact JSON with finite-number checks through existing helpers.
Existing bounded metadata traversal/serialization limits apply. This implementation
is for compact explicit scenes, not an unbounded graph archive.

## 12. Empty/sparse-state semantics

Zero objects, singleton objects, unrelated coexisting objects and motion-only input
can validly return no interactions, subsystems, assessments or transitions.
Unavailable-only interactions cannot create a subsystem. Skipped recognized sources
with missing references, singleton scope or unsupported status are audited.
Unrecognized signals are not interpreted. No output is fabricated to avoid emptiness.

`ComplexSystemsState` contains `complex_state_id`, aligned context,
`source_physical_state_id`, ordered `interactions`, `subsystems`,
`stability_assessments`, `transition_candidates`, `uncertainty`, `provenance` and
`schema_version`. Source snapshots and interaction/subsystem counts are audit only.

## 13. Why there is no complexity score

No global complexity, weighted stability percentage, entropy, emergence or
criticality score is defined. More edges or branches mean more represented source
structure, not stronger evidence or a more complex real system.

## 14. Why there is no causal discovery

Connected components carry no inferred direction, intervention semantics or causal
mechanism. No Granger causality, Bayesian network learning or causal discovery is
performed. Cascade scoring and propagation are deferred: shared endpoints do not
establish that a change propagates, how, or in what direction.

## 15. Why entropy/chaos/attractors are not inferred

These contracts provide categorical physical summaries, not the measured dynamical
system needed to justify entropy, Lyapunov exponents, attractors, chaos, criticality
or tipping-point probabilities. Variability and uncertainty are not substitutes.

## 16. Relationship to physics constraints

Current constraints and latent active summaries are reused without modification.
Shared constraint scopes and exact indicator triples are explicit engineering
rules. A violated image-plane assessment does not establish real-world breakdown.
Missing support geometry is not proof of instability.

## 17. Relationship to cross-modal consequences

Candidate modalities/statuses enrich existing exact-scope edges as provenance only.
Expected acoustic, tactile-force or thermal consequences remain candidates, not
heard sound, measured force or measured temperature. They cannot create coupling
or stability indicators by themselves.

## 18. Relationship to multiple futures

All supplied alternatives remain separate. Grounded contact, collision and
continuity patterns remain candidates, including unresolved alternatives. Relations
and branch counts are preserved in source snapshots without winner selection,
probability creation, posterior updating or confidence aggregation.

## 19. Relationship to Step 23 Connectome

Step 24 ends at ComplexSystemsState. The frozen connectome is unchanged. A later
explicit integration may add this as an optional routing source; no current
routing, reasoner execution or application wiring is added.

## 20. Relationship to Step 25 Topological Cognition

No topological cognition/TDA, persistent homology, Betti numbers or other
topological metric is implemented. Connectivity here is a deterministic grouping
operation over explicit candidate relations, not a topological cognition claim.

## 21. Relationship to Experiment 001

No improved prediction, hidden actor anticipation, TTA, complexity benefit,
cross-modal benefit, emergence or Experiment 001 success is demonstrated. Later
controlled ablations must compare fixed inputs and ground truth with and without
physics, cross-modal and systems-structure representation.

## 22. Explicit non-goals

No causal discovery, hidden actor inference, control/action policy, branch selection,
Bayesian update, ExperienceLearning mutation, current evidence promotion or scientific
validation. No frozen production, live_app, Mini-Lab or existing test changes. No
dependency, repository, model, dataset, GPU or internet use. Exactly three Step 24
files are added, with no staging, commit, push or tag.

## 23. Validation status

- 🧩 IMPLEMENTED
- 🧪 TESTED (synthetic engineering contracts and regressions)
- 📊 NOT YET TASK-LEVEL EVALUATED
- 🏆 NOT YET EXPERIMENTALLY SUPPORTED

Final combined run on 2026-09-15: **969 passed in 40.45 seconds**.

| Suite | Passed |
| --- | ---: |
| Step 24 dedicated | 55 |
| Step 23 connectome | 87 |
| Step 22 experience learning | 80 |
| Step 21 trajectory | 62 |
| Step 20 prediction bridge | 103 |
| Step 19 beliefs | 117 |
| Step 18 futures | 144 |
| Step 17 consequences | 93 |
| Physical world / constraints / latent / hybrid / common evidence | 228 |

No existing test was changed or weakened. Tests cover contracts, alignment,
independent optional inputs, grounded mapping, deduplication, components,
categorical assessment, transition grounding, provenance, uncertainty, immutable
serialization and no clock/feedback calls. These are engineering tests, not
task-level evidence of improved prediction or complex-systems benefit.

Reproduce the combined suite with the existing environment:

```powershell
.\.venv\Scripts\python.exe -m pytest evaluation/tests/test_complex_systems.py evaluation/tests/test_reasoning_connectome_v02.py evaluation/tests/test_experience_learning_v02.py evaluation/tests/test_astra_efa_experience.py evaluation/tests/test_belief_prediction_bridge.py evaluation/tests/test_bayesian_belief_state.py evaluation/tests/test_multiple_futures.py evaluation/tests/test_cross_modal_consequences.py evaluation/tests/test_physical_world_model.py evaluation/tests/test_physics_constraints.py evaluation/tests/test_latent_physical_state.py evaluation/tests/test_hybrid_world_state.py evaluation/tests/test_common_evidence_state.py -q
```

## 24. Canonical epistemic statements

> An interaction candidate is not a verified causal link.

> A connected component is a coupled subsystem candidate, not proof of a causal mechanism.

> Absence of violation is not positive evidence of stability.

> Instability is not equivalent to motion or uncertainty.

> A transition candidate is not an occurred event.

> Multiple possible futures are not dynamical attractors.

> Cross-modal expected consequences are not observations.

> Graph connectivity is descriptive, not causal.

> Interaction counts are audit information, not a complexity metric.

> Step 24 does not compute entropy, chaos, criticality, attractors or tipping-point probabilities.

> Step 24 does not infer hidden actors.

> Step 24 does not select a future branch.

> Step 24 does not update Bayesian beliefs.

> Step 24 does not modify ExperienceLearning.

> Step 24 is an engineering systems-representation layer, not scientific validation.
