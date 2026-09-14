# Multiple Futures v0.2 — Step 18

## 1. Purpose

Represent several conditional near-future continuations of the same explicit
physical state. Every branch exposes assumptions, structured changes, source
references and unresolved uncertainty. No branch is selected, ranked or assigned
a probability. This is an engineering candidate layer, not a validated predictor.

## 2. Architecture

```text
LatentPhysicalState + optional physics constraints/transition assessment
    -> existing HybridWorldState context checks
    -> explicit motion_state check
    -> existing CrossModalConsequenceBuilder prerequisite audit
    -> local source groups
    + optional supplied CrossModalConsequenceBundle references
    -> MultipleFuturesBuilder
    -> MultipleFutureBundle: branches + structural branch relations
    -> later belief/evaluation consumers (not implemented here)
```

The public Step 17 builder is reused to audit contact/discontinuity prerequisites
instead of duplicating its signal, endpoint and constraint-conflict rules. This
audit runs from the explicit physical state even when no consequence bundle is
supplied. Its internally generated consequences are not attached as supplied
evidence. Multiple modalities from one physical source are grouped before
branching; they are not independent votes or separate physical events.

The layer adds only `multiple_futures.py`, its dedicated test file and this
document. It changes no existing contracts, producer, Common Evidence integration
or live camera/photo/video behavior.

## 3. Input contracts

```python
from fifth_layer.world_model.multiple_futures import MultipleFuturesBuilder

builder = MultipleFuturesBuilder()
futures = builder.build(
    latent_state,
    physics_constraints=constraints,  # optional bundle or transition assessment
    consequences=consequence_bundle,  # optional actual Step 17 bundle
)
serialized = futures.to_json()

# An existing HybridWorldState may replace the first two arguments:
futures = builder.build(hybrid_state, consequences=consequence_bundle)
```

Inputs are `LatentPhysicalState` v0.1, optional `PhysicsConstraintBundle` or
`PhysicsTransitionAssessment` v0.2, and optional `CrossModalConsequenceBundle`
v0.2. A hybrid input and a separate constraint argument cannot be combined.
Learned signals pass existing hybrid validation but never drive generation or
change output ordering, branch identities or physical contents.

Constraints must match current scene/time and declared session/frame. Existing
recursive alignment rejects mismatched sessions/frames and future or unorderable
source timestamps. Consequence bundles must match all four context fields;
every candidate must reference the same latent state and current physical
objects. A declared bundle physical source must also match, including when the
bundle is empty. Bounded consequence metadata receives the same recursive
session/frame/time alignment check.

Transition assessments must retain
`retrospective_transition_ending_at_current_snapshot`. Their earlier timestamps,
assessment kind and provenance are preserved as source assessments. Step 18
does not reinterpret them as future predictions or reconstruct their history.

Physical relation/dynamics signals must already exist in the latent state. Bare
constraint status does not synthesize a trigger; use the existing latent builder
with the assessment when preparing inputs.

## 4. Output contracts

All contracts are frozen dataclasses, with detached immutable mappings/tuples
and deterministic `to_dict()`/`to_json()` methods.

`FutureStateCandidate` contains:

- `future_id`, scene/time/session/frame context, `horizon_kind`, `horizon_value`.
- `status`, `branch_family`, `object_ids`.
- `source_physical_state_id`, `source_constraint_ids`, `source_consequence_ids`,
  `source_field_references`.
- Structured `assumptions`, `predicted_changes`, string-tuple `uncertainty`,
  immutable `provenance`, `rule_id`.
- `confidence=None`, fixed `rule_version` and `schema_version` equal to
  `multiple-futures-0.2`.

Every branch requires nonempty identity, physical endpoints, source references
and assumptions. All non-None confidence values are rejected, including zero.
Status is bounded to `possible`, `indeterminate`, `unsupported`, `unavailable`.
The builder only emits the first two. `Unsupported` means prerequisites are not
supported, not impossible; `unavailable` means source state is missing, not false.

`MultipleFutureBundle` contains matching context, `source_physical_state_id`,
`branches`, `branch_relations`, `uncertainty`, `provenance` and fixed
`schema_version=multiple-futures-0.2`. It rejects duplicate branch identities,
duplicate semantic branches, mismatched source/context, dangling relation
references and duplicate relation pairs or IDs.

Predicted changes have exactly `field`, `value` and `epistemic_status`. Physical
changes use `branch_assumption`; consequence references retain `expected`,
`possible` or `indeterminate`. Unknown future motion/contact/continuity is the
explicit string `unresolved`; no numeric substitute is inserted. No arbitrary
trajectory, speed, sound, temperature or force field is accepted.

## 5. Branch families actually implemented

| Actual upstream signal | Branch assumptions | Conditional change |
| --- | --- | --- |
| Object attribute `motion_state` with value `moving`, status `observed` or `estimated`, and references | Motion persists; outcome remains unresolved | `motion_state=moving` or `unresolved` |
| `contact_candidate` from physical `contact_possible` | Contact within horizon; no contact within horizon; outcome unresolved | `contact_status=possible_contact`, `no_contact_assumed` or `unresolved` |
| `collision_risk_geometry` from constraint triple `collision_possibility / collision_possible / satisfied` | Same three contact alternatives | Same conditional contact changes |
| `tracking_or_motion_discontinuity` from `implausible_displacement / kinematic_outlier / violated` or `inertia_consistency / abrupt_change_detected / violated` | Sampled continuity resumes; discontinuity remains unresolved | `continuity_status=possible_continuity` or `unresolved` |

Motion persistence and contact/no-contact assumptions normally have status
`possible`; their unresolved alternatives have status `indeterminate`. Step 17
keeps tracking/camera ambiguity explicit, so both discontinuity alternatives
remain `indeterminate`. An indeterminate/conflicting constraint assessment also
keeps all affected contact alternatives indeterminate. Linked `unsupported` or
`not_applicable` constraints suppress their physical source group; neither
status becomes a certain future or a negative physical fact.

Step 17 consequence association is an addition to an existing physical branch,
not a separate speculative sensory story. Supplied candidates can attach to
the matching contact-compatible branch and to its unresolved alternative.
Discontinuity consequences attach to the unresolved branch. No-contact branches
do not inherit impact/contact consequences; continuity-resumption branches do
not inherit discontinuity-force consequences.

Source binding checks consequence identity, modality, event family, endpoints,
constraint IDs, rule/version, exact source signal and original source references.
Additional field detail from optional full constraints is accepted without
changing the consequence identity. Full supplied candidate records, including
their original status and uncertainty, are retained in branch provenance.
Unknown or unassociated candidate IDs are listed at bundle level; they do not
create new physical branches. Unsupported/unavailable consequences are not
attached. An absent bundle never blocks explicit physical futures.

No support/stability transition family is implemented: current
`support_candidate` describes static geometry, not load transfer or a transition.
Overlap, object class, disappearance, occlusion, boundary exit and learned
metadata alone generate no futures. There is no semantic glass/shattering rule.

## 6. Branch assumptions

`FutureAssumption` contains `assumption_id`, bounded `assumption_type`, nonempty
`source_refs` and an explicit `statement`. Its references must belong to the
branch source references. Types are `motion_persists`, `contact_within_horizon`,
`no_contact_within_horizon`, `continuity_resumes`, `outcome_unresolved`.

For example, `candidate geometry does not resolve to contact within the
represented horizon` is a branch assumption, not observed absence of contact.
Neither a changed categorical value nor assumption prose establishes reality.
The builder emits one explicit assumption per local branch. Duplicate assumption
IDs and contradictory contact assumptions within one branch are rejected.

## 7. Horizon semantics

Default: `next_transition`, `horizon_value=None`. Callers can explicitly select
qualitative `short_horizon`, also with `horizon_value=None`. Every numeric horizon
is rejected in v0.2 because no physical duration calibration is provided.

The output timestamp is the current source timestamp, never an invented target
or event time. Unknown source time remains `None` and adds
`temporal_alignment_unknown`. No exact displacement, collision time, future
position or acceleration magnitude is calculated, even if sampled pixel values
exist upstream. Source values remain accessible through physical provenance.

## 8. Branch relationship semantics

`FutureBranchRelation` contains a deterministic `relation_id`, exactly two
distinct `branch_ids`, bounded `relation` and explicit `basis`.

Only opposite `contact_within_horizon` and `no_contact_within_horizon` assumptions
for the same object pair and horizon are `mutually_exclusive`. Bundle context
validation also guarantees the same source state and source time. This relation
describes incompatible assumptions, not which assumption is correct. Different
source descriptions do not justify exclusivity by themselves.

All other generated pairs are `unresolved_relation` with basis
`joint_compatibility_not_evaluated`. The contract supports `co_possible`, but
this builder does not claim joint compatibility. The bundle rejects a
`co_possible` relation between opposite contact assumptions and rejects
unjustified exclusivity.

These are local hypotheses, not an exhaustive set of mutually exclusive global
world scenarios. The builder does not take a Cartesian product of object futures,
normalize outcomes or treat unresolved branches as exclusive physical events.

## 9. Provenance and determinism

Source physical state identity, context, endpoints, field references, linked
constraint IDs and supplied consequence IDs are explicit fields. Branch
provenance retains source details, original physical provenance, rule/schema
versions, assumption IDs and associated Step 17 records. The bundle additionally
retains optional constraint/assessment data and supplied consequence provenance.
Input objects are read only; neither producer is mutated.

Contact/discontinuity references preserve Step 17's semantic latent source
selectors and original field references. Motion uses the actual attributed
`motion_state` source plus a deterministic latent object field reference.
Rule IDs are `branch_motion_state`, `branch_contact_candidate`,
`branch_collision_risk_geometry`, `branch_tracking_or_motion_discontinuity`.

`stable_id` hashes semantic source identity: full source context, latent state ID,
qualitative horizon, branch family, endpoints, constraint/consequence references,
field references, structured assumption and rule/version. Assumption and relation
IDs are deterministic too. No random UUID or wall clock is used.

Ordering uses branch family, object IDs, assumption types and stable ID; relation
pairs are sorted by IDs. This is not preference or confidence ranking. Identical
source records are deduplicated by the existing source audit; distinct source
references can remain distinct representations without becoming independent
events. Conflicting duplicates are rejected.

Metadata uses existing `bounded_plain` limits: 50,000 nodes, depth 24, 16,384
characters per string and 1 MiB serialized structure. Opaque/raw payloads and
nonfinite values are rejected. No silent provenance truncation is performed.

## 10. Unknown/missing handling

Consistent missing-prerequisite policy: skip the unsupported source group.
For an eligible physical source, always include its unresolved alternative.
An entirely empty result records `no_eligible_explicit_branch_source`; absence
of a branch is never impossibility or evidence against an event.

Missing constraints and consequences are marked unavailable in bundle
uncertainty, not violated or false. Source state, object, constraint and
consequence uncertainty remains explicit. `None` never becomes zero or false.
Generated limitations include `future_is_hypothesis`, `branches_not_exhaustive`,
`horizon_duration_unknown`, `no_branch_is_not_impossibility` and
`metric_future_state_unknown`.

## 11. Explicit non-goals

No hidden-actor inference, observation claims, physical simulation, long-horizon
planning, semantic stories, calibrated probabilities, Bayesian updates, branch
weights, softmax, ranking, winning branch, belief formation or prediction scoring.
No supports/contradicts population or Common Evidence adapter. Provenance says
`truth_decision=not_performed`. No model/dependency installation, LLM, GPU, camera,
dataset, network access or external world-model repository is required.

## 12. Testing

Dedicated fixtures use actual physical dataclasses, latent builder, physics
engine and Step 17 builder. Tests cover contracts, nested immutability, bounded
changes/provenance, deterministic IDs/JSON across processes, input permutation,
source binding, optional constraint-detail interoperability, context rejection,
unknowns, consequence status preservation, no probability/observation/actor
claims, structural exclusivity and retrospective assessment provenance.

```powershell
.venv/Scripts/python.exe -m pytest evaluation/tests/test_multiple_futures.py -q
.venv/Scripts/python.exe -m pytest evaluation/tests/test_cross_modal_consequences.py -q
.venv/Scripts/python.exe -m pytest evaluation/tests/test_physical_world_model.py evaluation/tests/test_physics_constraints.py evaluation/tests/test_latent_physical_state.py evaluation/tests/test_hybrid_world_state.py evaluation/tests/test_common_evidence_state.py -q
```

Implementation verification: 144 Step 18 tests, 93 Step 17 tests and 228
world-model regression tests passed. Tests are local and deterministic. A fresh
process test also checks that generation imports none of torch, transformers,
ultralytics or cv2. No existing tests were weakened.

## 13. Validation status

- 🧩 IMPLEMENTED
- 🧪 TESTED
- 📊 NOT YET TASK-LEVEL EVALUATED
- 🏆 NOT YET EXPERIMENTALLY SUPPORTED

These results establish tested engineering behavior, not predictive accuracy or
physical realism.

## 14. Known limitations

Image-plane contact and motion remain ambiguous. Real depth, duration, metric
trajectory, material, force, acoustic emission and causal agents are unresolved.
Branches are local, deliberately narrow, non-exhaustive and not ranked. The
pairwise relation table grows quadratically; large outputs fail bounded metadata
validation rather than silently dropping branches. No scalability benchmark was
performed. The audit currently follows only Step 17's implemented physical
families plus explicit moving state; it is not a general rule language.

Full supplied consequence records are retained as source metadata, never verified
sensor facts. Their association requires current Step 17 semantic identity;
arbitrary manually named candidates remain unassociated. Source IDs and metadata
assume trusted upstream producers and do not authenticate observations. The layer
does not reconstruct source history or claim all compatible joint scenarios.

## 15. Relationship to Step 19 Bayesian Belief State

Step 18 supplies branch hypotheses, source references and assumptions for a later
Step 19 consumer. It does not create priors, likelihoods, posteriors, normalized
weights or Bayesian updates. Step 19 must not assume these local branches form
an exhaustive, disjoint probability space or select a winner based on their
deterministic order.

## 16. Relationship to Experiment 001

Step 18 provides structured near-future hypotheses that can later be evaluated
under **Vision + Physics** versus **Vision + Physics + Cross-Modal Consequences**.
It does not demonstrate hidden-actor anticipation, improved Time-to-Anticipation,
improved prediction accuracy or scientific support for AISTHESIS. Those require
later controlled evaluation and ground truth. No Experiment 001 success is claimed.

The existing Step 17 document and `evaluation/README.md` define related engineering
and experiment boundaries. Separate dedicated documents for the older latent,
physics, hybrid and Common Evidence contracts were not present under `docs`;
their actual source APIs and regression suites were inspected instead.

## 17. Canonical epistemic statements

> A possible future is not an observed fact.

> A future branch is a structured hypothesis, not reality.

> Multiple future candidates preserve uncertainty rather than resolving it.

> Absence of a generated branch does not prove impossibility.

> Branch ordering is deterministic, not probabilistic.

> Cross-modal consequences remain expected or possible consequences, not
> observed sensory evidence.

> Multiple Futures v0.2 does not infer a hidden actor.

> Multiple Futures v0.2 does not assign Bayesian probabilities.

> Multiple Futures v0.2 is an engineering candidate model, not a scientifically
> validated predictor.

Possible future != predicted fact. Plausible branch != reality. Most plausible
!= true. No branch != impossible. Branch ranking != calibrated probability.
Assumption != observation. Cross-modal consequence != event occurrence.
Constraint status != future certainty. Missing state != negative evidence.
Learned representation != physical fact. One branch != exclusive truth.
Multiple branches != multiple real events.
