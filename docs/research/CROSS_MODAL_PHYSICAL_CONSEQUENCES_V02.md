# Cross-Modal Physical Consequences v0.2 — Step 17

## 1. Purpose

Represent conditional sensory consequence candidates from explicit physical
summaries, preserving source uncertainty. The output describes what might follow
if the physical interaction is real. It supplies neither observations nor event
confirmation. This is an engineering candidate layer.

## 2. Architecture

```text
PhysicalWorldState + physics assessments
    -> existing LatentPhysicalStateBuilder
    -> LatentPhysicalState (+ optional matching physics assessments)
    -> CrossModalConsequenceBuilder
    -> immutable CrossModalConsequenceBundle
    -> future consumer (not implemented here)
```

`HybridWorldState` is also accepted. Its existing context and schema validation
is reused for both entry points. Learned metadata never drives generation.
No existing producer, live UI, sensing behavior, or Common Evidence contract is
changed. The bounded serialization helper in `common_evidence_state.py` is reused
without adding an evidence adapter or inserting items into Common Evidence State.

## 3. Input contracts

```python
from fifth_layer.world_model.cross_modal_consequences import CrossModalConsequenceBuilder

builder = CrossModalConsequenceBuilder()
consequences = builder.build(latent_state, physics_constraints)
# Alternatively: consequences = builder.build(hybrid_state)
serialized = consequences.to_json()
```

`latent_state` must be `LatentPhysicalState` v0.1. The optional second argument is
`PhysicsConstraintBundle` v0.2 or `PhysicsTransitionAssessment` v0.2. Supplying
both a hybrid container and a second constraint argument is rejected.

The explicit latent relations and dynamics are the rule triggers. Optional
constraints audit referenced source summaries and preserve detailed provenance;
they do not independently generate candidates. Call the existing latent builder
with the assessments first to obtain its recognized relation/dynamics signals.
This avoids treating a bare constraint status as an event.

Scene and timestamp must match between the state and supplied constraints.
Declared session, coordinate frame and nested source timestamps obey existing
HybridWorldState validation: mismatches and future/unorderable sources are
rejected. Unknown snapshot time remains `None`. Relations require two current
object endpoints; dynamics referring to absent objects are skipped. Linked
constraint endpoints must match the candidate endpoints, or construction fails.

Transition assessments remain
`retrospective_transition_ending_at_current_snapshot`; other assessment kinds
are rejected. Their IDs, kind and provenance are retained. The consequence
builder does not reconstruct history, revalidate historical observations or
assign a future event time. The timestamp identifies source context only.

## 4. Output contracts

`PhysicalConsequenceCandidate` is a frozen dataclass containing:

- Identity and context: `consequence_id`, `scene_id`, `timestamp`, `session_id`,
  `coordinate_frame_id`.
- Consequence: `modality`, `event_family`, `status`, `object_ids`, `description`.
- Traceability: `source_physical_state_id`, `source_constraint_ids`,
  `source_field_references`, `rule_id`, `provenance`.
- Uncertainty: canonical string tuple `uncertainty`, and `confidence=None`.
- Fixed `schema_version` and `rule_version`: `cross-modal-consequences-0.2`.

`CrossModalConsequenceBundle` contains matching context, `candidates`,
`provenance`, `uncertainty` and fixed `schema_version` with the same version.
Candidate context must match the bundle, including coordinate frame and unknown
time. Duplicate IDs and duplicate semantic source candidates are rejected.

All five statuses and four modalities are bounded string enums. Every non-None
confidence is rejected, including zero. Neither contract contains an observation
flag, supports/contradicts fields, a belief score or a truth decision result.
Generated provenance states `truth_decision=not_performed`.

Mappings are detached and recursively frozen; sequence fields become tuples.
`to_dict()` returns detached JSON data and `to_json()` uses sorted keys, compact
separators and rejects nonfinite numbers. Bundle ordering is modality, event
family, object IDs, then consequence ID.

## 5. Implemented rule families and actual signals

Inspection used the existing physical, constraint, latent, hybrid and Common
Evidence implementations and their dedicated tests. `evaluation/README.md`
provides the existing Experiment 001 boundary. Separate documentation for the
five older contracts was not present under `docs`; no substitute APIs were
invented.

| Actual latent signal | Actual upstream source | Consequence outputs | Generated status |
| --- | --- | --- | --- |
| `contact_candidate` | Physical relation `contact_possible`, retained as `possible` by the latent builder | tactile_force: `sustained_contact_force` | possible |
| `collision_risk_geometry` | Exact constraint triple `collision_possibility`, `collision_possible`, `satisfied` | acoustic: `impact`; tactile_force: `contact_impulse`; kinesthetic: `abrupt_motion_change` | possible |
| `tracking_or_motion_discontinuity` | Exact triple `implausible_displacement`, `kinematic_outlier`, `violated`, or `inertia_consistency`, `abrupt_change_detected`, `violated` | kinesthetic: `acceleration_change`; tactile_force: `inertial_force_change` | indeterminate |

Rule identifiers are `conditional_` followed by the exact latent signal name.
Relation values must be literal `True` or `possible`, with `possible` or
`indeterminate` source status and nonempty references. Constraint-derived signals
also require a linked active summary or supplied result. Multiple linked
assessments are retained without confidence pooling.

Contact alone supplies no duration or load: `sustained_contact_force` is only a
conditional possibility if loaded contact persists. It does not produce sound,
impulse or heat. Collision geometry indicates image-plane overlap/contact plus
approach; actual depth, contact, restitution and acoustic emission remain unknown.
The three collision outputs describe modalities of a candidate interaction,
not three confirmed events.

Motion discontinuity can be caused by tracking error or camera motion, so neither
physical acceleration nor inertial force is established. No measured acceleration
or force magnitude is output. Constraint violation alone never identifies cause.

When a linked source assessment conflicts with the recognized triple or is
indeterminate, the consequence becomes `indeterminate`. Linked `unsupported` or
`not_applicable` assessments suppress generation. They do not prove nonoccurrence.
Identical source records are deduplicated; conflicting copies with the same
semantic identity are rejected instead of resolved by input order.

There are no friction, heating, deformation, breakage, or support-transition rules.
The existing `support_candidate` is static support geometry, not a load-transfer
transition. The thermal modality is represented in the contract but no thermal
candidates are generated. Mere `overlaps`, class labels, object presence, sampled
continuity, disappearance and occlusion do not trigger these rules.

## 6. Status semantics

| Status | Meaning |
| --- | --- |
| expected | A normal predicted consequence given explicit prerequisites and stated assumptions; still not observed. Reserved by this builder because current signals are too weak. |
| possible | Physically plausible conditional consequence with insufficient constraints for expected. |
| unavailable | Required source state is missing; never false. Valid contract status; the builder instead skips missing prerequisites. |
| unsupported | Available state does not support prerequisites; never proof that an event did not happen. Valid contract status; the builder skips these sources. |
| indeterminate | Ambiguous or conflicting source information prevents stronger classification. |

The builder only emits `possible` and `indeterminate`. There is no `observed`
status and no fabricated probability.

## 7. Provenance semantics

Candidate provenance retains physical state identity/schema/provenance, context,
source section, exact signal/status, upstream references, rule identity/version
and relevant constraint summaries. Supplied detailed constraint results retain
their own IDs, rule versions, field references, uncertainty and provenance.
Without full results, the existing active summaries supply constraint identity
and vocabulary; missing details are not reconstructed.

Field references include upstream `derived_from` and a semantic selector of the
form `<latent_state_id>.<relations|dynamics>[<latent-source hash>]`. This selector
is computed from section, signal, endpoints and upstream references; it is not a
numeric array index or an invented upstream relation ID. The corresponding
source section, signal and references are also recorded in provenance.

IDs use the existing `stable_id` helper over rule version, full context, latent
state ID, modality, event family, sorted object/constraint IDs, upstream references
and rule identity. No UUID or wall clock is used. Different source references
remain distinct candidates; these are not independent real events or independent
evidence votes. IDs identify semantic sources, not mutable assessment scores.

State, relevant object and source uncertainties propagate as strings. Supplied
constraint uncertainty also survives on the output bundle when no rule fires.
The builder adds explicit limitations rather than numeric certainty.

The existing bounded metadata traversal imposes 50,000 nodes, depth 24, 16,384
characters per string and a 1 MiB serialized ceiling per checked structure.
Opaque payloads, raw embeddings and nonfinite numbers are rejected. Oversized
provenance fails rather than being silently truncated.

## 8. Unknown and missing handling

Missing prerequisites consistently skip generation. An empty bundle means that
no eligible source candidate was represented; it says nothing about actual
events, audible sound or sensor availability. `None` is never converted to zero
or false. Unknown time adds `temporal_alignment_unknown` and remains unknown.

Every bundle declares `missing_prerequisite_policy=skip_not_negative_evidence`.
Every generated candidate carries `consequence_not_observation`,
`event_occurrence_unconfirmed`, `sensor_availability_not_evaluated` and
`metric_magnitude_unknown`. This module does not inspect microphones, force
sensors, temperature sensors or body-state sensors.

## 9. Explicit non-goals

No sensing, hidden-actor inference, event confirmation, learned representation
interpretation, belief formation, evidence support/contradiction assignment,
calibrated probability, physical simulation or metric force/temperature/audio
prediction. No integration into live camera/photo/video paths. No model,
dependency, network access, external world-model repository or LLM is added.

## 10. Testing

All fixtures are synthetic and local. They use the actual immutable dataclasses,
the existing latent builder, and end-to-end physics engine fixtures for collision
geometry and motion change. Tests cover validation, immutability, bounded
metadata, deterministic JSON and IDs across processes, missing data, source
conflicts, context protection, provenance, learned-signal isolation, candidate
separation and absence of observation/belief claims. A subprocess check verifies
that building candidates imports none of torch, transformers, ultralytics or cv2.

Run from the repository root:

```powershell
.venv/Scripts/python.exe -m pytest evaluation/tests/test_cross_modal_consequences.py -q
.venv/Scripts/python.exe -m pytest evaluation/tests/test_physical_world_model.py evaluation/tests/test_physics_constraints.py evaluation/tests/test_latent_physical_state.py evaluation/tests/test_hybrid_world_state.py evaluation/tests/test_common_evidence_state.py -q
```

Final verification passed 93 dedicated tests and 228 existing regression tests:
321 tests total in the combined run.

## 11. Validation status

- 🧩 IMPLEMENTED
- 🧪 TESTED
- 📊 NOT YET TASK-LEVEL EVALUATED
- 🏆 NOT YET EXPERIMENTALLY SUPPORTED

Tests establish engineering behavior for the tested contracts and fixtures.
They do not validate the physical realism or usefulness of sensory predictions.

## 12. Known limitations

Source geometry is image-plane geometry. Depth, mass, material, actual contact,
sound propagation, contact duration and real motion causes are unresolved.
No expected-status rule is justified yet. Kinesthetic candidates describe
conditional object-motion consequences, never measured body state. No thermal
rule is available. Latent summaries lack complete source detail unless matching
constraint results are supplied. Inputs must be trusted explicit producer
contracts: arbitrary manually authored text in provenance is retained as source
metadata, not verified or interpreted as physical evidence.

There is no prediction horizon, event-time estimator, history reconstruction,
sensor comparison, task scoring or Common Evidence adapter. Consumers must not
count candidates as occurrences or treat distinct source representations as
independent corroboration.

## 13. Relationship to Experiment 001

Step 17 creates a representation needed for a later comparison of
**Vision + Physics + Cross-Modal Consequence** versus **Vision + Physics**.
It does not demonstrate Experiment 001's hypothesis. Earlier hidden-actor
anticipation, improved Time-to-Anticipation (TTA), improved prediction accuracy
and experimental support require the later benchmark with appropriate ground
truth and controlled comparisons. None is claimed here.

## 14. Canonical epistemic statements

> Expected sensory consequence is not observed sensory evidence.

> Physical plausibility is not event occurrence.

> Missing modality is not negative evidence.

> A candidate acoustic consequence does not mean a sound was heard.

> A candidate tactile-force consequence does not mean force was measured.

> Cross-modal consequence generation does not infer a hidden actor.

> Constraint satisfaction or violation is not itself a sensory event.

> Cross-modal Physical Consequences v0.2 is an engineering candidate model,
> not a scientifically validated sensory simulator.

Expected != observed. Possible != occurred. Acoustic consequence != heard sound.
Tactile/force consequence != measured force. Thermal consequence != measured
temperature. Kinesthetic consequence != measured body state. Physics plausibility
!= reality. Missing sensor != negative evidence. Constraint status != event
occurrence. Learned representation != physical fact. Multiple candidates !=
multiple real events.
