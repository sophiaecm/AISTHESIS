# Physics Constraint Engine v0.2

## Purpose and scope

An opt-in, model-agnostic layer of deterministic constraints over the frozen
Physical World Model v0.1. It assesses sampled image-plane observations and
transitions without changing physical state or production behavior. All code is
in new modules; the v0.1 builder, models, evaluator and documentation are unchanged.

This is not a simulator, a detector, a reasoner replacement, or a learned dynamics
model. It does not authenticate measurements or establish physical truth. It
does not use VLM text, experience retrieval, expected sensory consequences, neural
models or online learning. No metric calibration, mass, force, friction, energy,
momentum, depth or mechanical stability is inferred.

## API

```python
from fifth_layer.world_model.physics_constraint_engine import (
    PhysicsConstraintEngine, PhysicsConstraintPolicy,
)

policy = PhysicsConstraintPolicy(
    max_displacement_pixels=100.0,
    max_velocity_change_pixels_per_second=200.0,
    support_gap_pixels=0.0,
)
assessment = PhysicsConstraintEngine(policy).assess(
    current_physical_state,
    history=(earlier_physical_state, previous_physical_state),
    session_id='one-tracking-session',
    coordinate_frame_id='one-fixed-image-coordinate-frame',
)
serialized = assessment.to_json()
```

Input is the existing immutable `PhysicalWorldState`, never hypothesis text.
History contains chronological adjacent snapshots, including frames with no
objects. The caller must supply a single tracking session and unchanged image
coordinate frame. Both identifiers are mandatory and retained in provenance.
The engine cannot discover an omitted intermediate snapshot or independently
verify a camera coordinate change from the v0.1 schema.

`PhysicsConstraintResult` has controlled status, finding, explanation, measured
values, configured expected range, exact field references, uncertainty and rule
version. `PhysicsConstraintBundle` sorts results. `PhysicsTransitionAssessment`
records the current and previous scene/time and explicitly labels the evaluation
as a retrospective transition ending at the current snapshot. Results are frozen,
with recursively detached immutable mappings. Confidence is always null: this
version assigns no calibrated constraint probabilities.

## Epistemic status model

| Status | Meaning |
| --- | --- |
| satisfied | The stated image-plane predicate is met; not a confirmed physical event |
| violated | An explicitly configured heuristic range is exceeded; not physical impossibility |
| indeterminate | Measurements, timing, association or interpretation are insufficient |
| unsupported | Measurements exist but the requested heuristic threshold is unconfigured |
| not_applicable | The tested geometry predicate or prerequisite is absent |

The `finding` gives the specific interpretation. For example, `collision_possible`
with status `satisfied` means the possibility predicate was satisfied, never that
a collision happened. `collision_geometry_absent` refers only to sampled 2D
geometry; it does not exclude an event between frames. Constraint results never
write attributes back to physical objects and never instantiate hidden objects.

Only attributed `observed` or `estimated` pixel centers/bboxes from objects with
`visibility_state='observed'` are accepted as measurements. `possible` geometry,
missing units and absent provenance do not establish measured motion. The existing
builder's bbox-derived centers are accepted estimates. Occlusion possibility is
read separately and retains its possibility interpretation. Unknown values stay
null, including missing occlusion evidence and velocity. Source statuses and
provenance are trusted input contracts, not independently verified sensor truth.

## Constraint families

1. **Kinematic continuity:** same typed track ID across adjacent snapshots;
   displacement = current center minus previous center. Velocity requires both
   pixel centers and a positive source time interval and is labeled pixels/second.
   A satisfied result verifies a sampled transition is available, not the path
   between samples. No motion is fabricated when IDs change or a track reappears
   after a gap. Stored v0.1 velocity is not substituted for missing measurements.
2. **Inertia consistency:** three consecutive same-track observations establish
   two pixel-velocity vectors. Their Euclidean difference is compared with an
   explicit heuristic bound. `abrupt_change_detected` is a review finding only;
   it establishes no cause, collision, force or impulse.
3. **Object permanence:** compare observed track sets. A missing track with prior
   truncation yields `possible_frame_exit`; prior overlap/occlusion geometry can
   yield `possible_occlusion`; otherwise `unexplained_track_loss`. All missing-track
   interpretations are indeterminate. A reobserved ID is labeled `reobserved_track`,
   not retroactively confirmed occlusion. Earlier object IDs are audit references,
   not current hidden-object entries.
4. **Support/stability possibility:** directional bbox test, upper center above
   lower center, positive horizontal overlap, and upper bottom within the lower
   vertical extent (allowing the configured gap). Results remain
   `support_geometry_possible`, `support_geometry_not_present`, or `indeterminate`.
   This broad image-plane predicate does not establish contact or stability.
5. **Collision possibility:** two observed tracks have decreasing center distance
   between adjacent timed snapshots and currently touching/intersecting bboxes.
   Only then is `collision_possible` emitted. Overlap alone stays indeterminate.
   No continuous swept-volume/contact solver is attempted.
6. **Frame/occlusion consistency:** interprets the coexistence of observed
   availability, supplied truncation, observed overlap, loss and reobservation.
   Loss remains uncertain even with compatible geometry. This family does not
   independently recompute frame boundaries: image dimensions are not first-class
   fields in v0.1 physical snapshots. It cannot confirm occlusion or exit.
7. **Implausible displacement:** raw same-track pixel jump compared with an
   explicit heuristic bound. `kinematic_outlier` may indicate identity mismatch,
   jitter or a large legitimate projected movement. It is not physical impossibility.

## Thresholds and image-plane limitations

The engine defaults both outlier thresholds to **None**, so it reports measurements
without inventing a plausibility bound. The support gap defaults to **0 pixels**,
meaning exact boundary contact or overlap under the documented bbox predicate.
Approach uses strictly decreasing center distance with no noise deadband.
All thresholds are finite, nonnegative and exposed through `PhysicsConstraintPolicy`.

The controlled offline reports explicitly select **100 pixels per adjacent
snapshot** for jump review and **200 pixels/second velocity-vector change** for
inertia review. These are demonstration heuristics selected for inspectability,
not calibrated or scientifically universal limits. The saved samples have 0.5-second
spacing. Different frame rates, resolutions, perspectives and camera motion can
require different policies. The thresholds and their heuristic role appear in
every result. Raw displacement/velocity are preserved, never clipped or zeroed.

No conversion to meters, m/s, kilograms, Newtons or m/s² occurs. Stationary camera
assumptions, detector jitter, uncertain identity, perspective and depth remain
unresolved. Small apparent shifts in overlapping parked objects can produce
collision possibilities; this deliberately weak predicate is not an event detector.

## Provenance, determinism and time

Every result references inspected physical-state attribute paths and retains
their original `derived_from` references. Missing fields still have an inspected
path, so indeterminate results remain auditable. Temporal measurements reference
both snapshot timestamps. Policy, session, coordinate frame and epistemic role
are attached to every result. Missing input attribution is not manufactured;
it leaves the corresponding measurement unavailable. Malformed v0.1 records
with unattributed non-null attributes are rejected by the existing contract.

Object/result ordering is stable; IDs are SHA-256 content identifiers. No random
IDs or wall-clock timestamps are generated. Canonical JSON is deterministic for
the same typed inputs, including the same numeric representation of policy values.

Known timestamps must be strictly increasing; duplicate scenes and future/reordered
history are rejected. Nested timestamp fields in physical source provenance are
validated against their own snapshot time. Results are issued at the ending
snapshot and cannot revise earlier assessments. The offline utility preserves
stored order and never silently sorts a future frame into history. An undated
pair may report center displacement based on caller-asserted sequence order,
but cannot establish velocity, inertia or approach. Dated history with an undated
current snapshot is rejected. Unverifiable source-reference strings cannot prove
when external measurements were actually made; source contracts remain necessary.

## Offline evaluation

```powershell
.\.venv\Scripts\python.exe -m evaluation.physics_constraints results_case_a_physical_world.json --output results_case_a_physics_constraints.json --max-displacement-pixels 100 --max-velocity-change-pixels-per-second 200
.\.venv\Scripts\python.exe -m evaluation.physics_constraints results_case_b_physical_world.json --output results_case_b_physics_constraints.json --max-displacement-pixels 100 --max-velocity-change-pixels-per-second 200
```

Only stored `physical-world-evaluation-0.1` data is read. The evaluator rehydrates
the existing contracts without invoking the physical-state builder, perception,
tracker, VLM, reasoners or camera. Sessions remain separate. Source SHA-256 is
retained and output files use exclusive creation: existing results cannot be
overwritten. Reports contain assessments and summaries, not new physical objects.

| Check | Case A | Case B |
| --- | ---: | ---: |
| Snapshots | 20 | 10 |
| Source sports-ball displacement observations | 8 | 3 |
| First observed person | 3.0 s | none |
| Constraint results | 1,945 | 842 |
| Jump heuristic violations | 4 | 2 |
| Inertia heuristic violations | 0 | 0 |
| Collision possibilities (not observed collisions) | 25 | 13 |
| Support geometry possibilities | 31 | 12 |

Case A jumps: ball at 3.5 s (109.762 px) and 4.0 s (102.650 px); person at
4.5 s (126.613 px) and 5.0 s (124.418 px). Case B jumps: ball at 3.5 s
(110.800 px) and 4.0 s (101.951 px). These exceed the selected 100-pixel bound;
they do not diagnose a tracking error or an impossible trajectory.

Source hashes and deterministic replay were checked. All assessment object IDs
were verified against observations at or before the assessment timestamp. Case A
introduces no person before 3.0 s; Case B introduces no person. Neither creates
hidden actors, confirmed collisions/contact or physical parameter estimates.
Aggregate finding counts combine families: permanence and frame/occlusion may
report the same track availability finding separately, not independent events.

## Tests and regression

45 new unittest tests cover all 30 requested categories, plus reappearance gaps,
typed track IDs, explicit/absent policies, wrong units, possible geometry exclusion,
immutability, strict deserialization, deterministic offline replay and CLI overwrite
protection. **506 automated tests passed: 461 baseline + 45 new**, using the existing
`.venv`. No dependencies were installed. The full automated suite comprises 25
unittest modules in the root and `evaluation/tests`; root executable model demos
without unittest cases were excluded to avoid model inference/downloads, as in
the baseline validation. The test command and selection method are documented
in the frozen v0.1 documentation; the same AST-based unittest selection was used.

## Next architectural step

A future Latent Physical State can consume these attributed measurements and
constraint assessments as separate channels, preserving unknowns and provenance.
Learned evidence should enter through a versioned, explicitly labeled evidence
contract rather than replace constraints or upgrade possibilities into facts.
V-JEPA and JEPA-WM sandboxes remain later work. This engine has no imports,
dependencies or assumptions tied to any learned model architecture.
