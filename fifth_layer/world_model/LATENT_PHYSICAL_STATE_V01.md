# Latent Physical State v0.1

## Purpose and architecture

This is a compact, deterministic **non-neural structured summary**, not an
embedding or a hidden-object model. It aligns the frozen Physical World Model
v0.1 with Physics Constraint Engine v0.2 evidence. Neither source is replaced,
rerun, modified or connected to the production pipeline.

```text
PhysicalWorldState + PhysicsConstraintBundle / PhysicsTransitionAssessment
    -> LatentPhysicalStateBuilder
    -> immutable LatentPhysicalState
    -> optional structured feature dictionary for a future adapter
```

Only five implementation/documentation/test files are added, plus two untracked
offline reports. There are no new dependencies, models, neural training, downloads,
online learning, embeddings, JEPA interfaces with tensor assumptions, or production
behavior changes.

## Schema and API

```python
from fifth_layer.world_model.latent_physical_state_builder import LatentPhysicalStateBuilder

latent = LatentPhysicalStateBuilder().build(
    physical_state,
    physics_assessment,  # alternatively a matching bundle, or None
    history=previous_latent_states,
    session_id='one-tracking-session',
    coordinate_frame_id='one-fixed-pixel-frame',
)
canonical_json = latent.to_json()
features = latent.to_feature_dict()
```

`LatentPhysicalState` is a frozen dataclass with recursively immutable nested
records. Object, relation and dynamics records are read-only mappings rather
than additional model classes, keeping the new schema small. All structures
are detached from caller-owned mappings. `to_dict()` returns a detached JSON view.

Schema `latent-physical-state-0.1` contains:

* `latent_state_id`, `scene_id`, `timestamp`, `session_id`, `coordinate_frame_id`.
* `objects`: current physical object ID, attributed state fields and uncertainty.
* `relations`: current endpoint IDs, signal, epistemic status, value and references.
* `dynamics`: temporal signal, epistemic status, referenced object IDs and evidence.
* `active_constraints`: compact IDs, families, original statuses/findings and IDs
  of their subject objects. This includes indeterminate/unsupported/not-applicable
  evaluations, not just positive signals.
* `uncertainty`: sorted flags, including source constraint uncertainties.
* `provenance`: physical scene, assessment, previous latent-state reference,
  original physical uncertainty metadata and contextual evidence IDs.

Objects retain only track identity, observed class, center, bbox, size, truncation,
displacement, pixel velocity, motion state/direction, visibility and possible
occlusion. Each field retains `value`, `status`, `units`, original `derived_from`
and an exact source attribute path. Rules are available at the referenced physical
record rather than copied as repeated descriptions. Missing fields remain null.
Non-observed class estimates are left unknown. No mass, force, friction, depth,
3D pose, energy, momentum or metric physics fields are synthesized.

Only the current physical object's IDs create object records. Past IDs may appear
in dynamics/audit references during a gap, but do not instantiate absent objects.
Relation endpoints must both exist in the current snapshot.

## Epistemic separation and constraint mapping

Physical attributes preserve `observed`, `estimated`, `possible`, `unknown` and
`unavailable`. Compression does not upgrade these roles. Accepted image-plane
units remain `pixels` or `pixels/second`; unsupported units leave the relevant
feature unknown and add an uncertainty flag.

All original constraint statuses (`satisfied`, `violated`, `indeterminate`,
`unsupported`, `not_applicable`) and findings remain in the compact evidence list.
Mapping uses exact **family + finding + status** combinations:

| Constraint evidence | Latent signal | Status |
| --- | --- | --- |
| collision_possibility / collision_possible / satisfied | collision_risk_geometry | possible |
| support_stability_possibility / support_geometry_possible / satisfied | support_candidate | possible |
| implausible_displacement / kinematic_outlier / violated | tracking_or_motion_discontinuity | possible |
| inertia_consistency / abrupt_change_detected / violated | tracking_or_motion_discontinuity | possible |
| inertia_consistency / motion_continuity_consistent / satisfied | sampled_motion_continuity | estimated |
| object_permanence / possible_frame_exit / indeterminate | boundary_exit | possible |
| object_permanence / possible_occlusion / indeterminate | occlusion | possible |
| object_permanence / unexplained_track_loss / indeterminate | disappearance | possible |
| object_permanence / reobserved_track / satisfied | reobserved_object | estimated |
| object_permanence / object_permanence_consistent / satisfied | permanence_compatibility | estimated |

Support direction comes from the explicitly stored upper/lower endpoint IDs,
which must agree with the constraint's current endpoints. Existing physical
contact/support possibility relations remain candidates. Existing left/right/
above/below/overlap predicates retain their source statuses; overlap is never
turned into collision. New IDs relative to supplied history yield a
`newly_observed_object` signal referencing current and prior snapshots. A first
snapshot has no inferred prior appearance history. Reobserved constraints do not
turn previous gaps into confirmed occlusion.

Constraint explanations, raw measured arrays, expected ranges and full provenance
are not duplicated. Constraint IDs resolve these at the source report. Semantic,
experience and expected sensory metadata are not used to create signals. Their
existing evidence IDs can remain contextual references without strengthening any
physical field. The implementation trusts typed source contracts; it cannot
authenticate whether an upstream producer falsely labels an observation.

## Temporal alignment and source matching

Every bundle/assessment must match the physical scene ID and timestamp. Every
constraint must match the supplied session and coordinate frame. Unknown actor
IDs are rejected. Assessment previous-scene/time and its full history scene-ID
list must exactly match supplied latent history.

History must have strictly increasing **known** timestamps, all earlier than the
current timestamp. Duplicate scenes, equal timestamps, future history, reordered
history, and cross-session/frame history are rejected. A single undated snapshot
is allowed without history. Nested timestamp fields in physical/constraint source
provenance and constraint measurements are checked against current time.

History is not used to fabricate motion or positions. The layer trusts the frozen
source fields and summarizes transition evidence at the current ending snapshot.
Opaque provenance reference strings cannot independently authenticate measurement
time; correct source contracts and stable tracking/frame identities remain caller
responsibilities. No wall-clock or random IDs are used. Canonical content hashes,
sorted records and deterministic serialization give stable output for identical
typed inputs. CLI files use the host platform's text newline convention.

## Uncertainty

Every object and state has explicit unresolved depth, metric scale, actual contact,
events between samples, tracking identity, occlusion, hidden state, support
stability and collision confirmation flags. Missing/unavailable attributes add
field-specific flags. Source constraint uncertainty is retained globally and for
its current object endpoints. Original physical uncertainty metadata is preserved
separately from derived signals. There is no uncertainty-to-confidence aggregation
or invented calibrated probability.

Unknown is not absent, zero, false or contradiction. A source `frame_truncation`
value of false can remain false because it is supplied evidence; missing truncation
stays null. Possibility signals never establish actual contact, support or collision.

## Future learned-latent compatibility

`to_feature_dict()` returns schema `latent-physical-features-0.1`, a detached,
versioned numeric/categorical dictionary. It does not return a neural vector or
assume any architecture, tensor dimension, normalization, JEPA model or calibration.

Per-object numeric fields expose value, units, source status and component-wise
`validity_mask`. Unknown values remain null with false masks; a valid measured or
estimated zero remains zero with true masks. Possible geometry retains its value
and possibility status but has a false measurement-validity mask. Categorical
fields also retain source status. Counts describe explicitly represented objects
and signals, not confirmed physical events. A zero collision-possibility count
does not mean collisions were physically excluded.

A later adapter can compare this explicit view with a learned latent state while
keeping measurement validity and epistemic roles separate. Learned evidence should
not replace the explicit contracts or silently overwrite uncertainty.

## Offline utility and controlled checks

```powershell
.\.venv\Scripts\python.exe -m evaluation.latent_physical_state results_case_a_physical_world.json results_case_a_physics_constraints.json --output results_case_a_latent_physical_state.json
.\.venv\Scripts\python.exe -m evaluation.latent_physical_state results_case_b_physical_world.json results_case_b_physics_constraints.json --output results_case_b_latent_physical_state.json
```

The utility only deserializes stored reports and invokes the new builder. It does
not invoke the physical-state builder, constraint engine, detector, tracker, VLM,
reasoners or camera. Physical and constraint session sets and snapshot counts must
match; source ordering is preserved, never silently repaired. The constraints'
source-file SHA-256 must match the exact physical input bytes. Both input file
hashes are recorded in the output report, making each referenced record traceable
to the original saved artifact. Output must be a new `*_latent_physical_state.json`
file; exclusive creation prevents overwrite.

| Check | Case A | Case B |
| --- | ---: | ---: |
| Snapshots | 20 | 10 |
| First person representation | 3.0 s | none |
| Sports-ball displacement records | 8 | 3 |
| Collision geometry possibilities | 25 | 13 |
| Support candidates | 31 | 12 |
| Latent output bytes | 4,189,851 | 1,831,989 |
| Combined physical + constraint source bytes | 12,466,925 | 5,404,835 |
| Output/source size ratio | 33.61% | 33.90% |

Current object-ID sets and timestamps exactly match their physical snapshots.
Case A introduces no person before 3.0 seconds; Case B introduces no person at all.
No hidden actors or confirmed collision/contact/support facts are generated.
Ball displacement and velocity preserve image-plane units. Every snapshot retains
explicit uncertainty. Offline replay matched the saved output bytes under the same
serialization convention, and source hashes were verified. These are architectural
compression checks, not improved physical accuracy claims.

## Validation and limitations

**52 new tests; 558 total automated tests passed** (506 frozen baseline + 52 new),
using the existing `.venv`, across 26 unittest modules. Tests cover all requested
boundaries, including missing-value masks, source statuses, temporal ordering,
constraint/source matching, current-only objects, immutable output, SHA mismatch,
and exclusive output creation. The offline unit test mocks both previous builders
to raise if invoked. Root executable inference demos without unittest cases are
excluded, following the frozen baseline's full automated-suite selection; no
models or dependencies were installed or rerun.

This is structured compression, not a minimal numeric embedding. Full audit paths
and geometric relations still take space; constraint raw records must remain
available for detailed inspection. No new physical threshold, motion estimate or
calibration is introduced. A source outlier or weak collision possibility remains
weak; compression does not resolve camera motion, jitter, identity errors, depth
or sampling gaps. Histories must use the same complete sequence as a supplied
assessment. General neural fusion, metric/3D state, independent source verification
and arbitrary asynchronous sensors remain later work.
