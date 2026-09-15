# Spatial Grounding Adapter v0.1

## 1. Purpose

A grounding request is a request for spatial localization evidence, not an observation itself.

Step 26B creates an isolated, backend-neutral boundary from frozen Step 26A symbolic targets to caller-supplied localization results. It neither runs inference nor reads images. Output schema is `spatial-grounding-v0.1`.

## 2. Position in AISTHESIS

The data flow is `ObservationRequestPlan + target_id + GroundingFrame -> GroundingRequest`, then an external boundary, then `RawGroundingObservation -> GroundedRegion -> GroundedObservationTarget`. The only new executable class is `SpatialGroundingAdapter`. Five frozen records accompany it: `GroundingFrame`, `GroundingRequest`, `RawGroundingObservation`, `GroundedRegion`, and `GroundedObservationTarget`.

## 3. Step26A symbolic target -> grounding request

`create_request(plan, target_id, frame, requested_output_types=('box', 'point'))` requires an actual `ObservationRequestPlan` with schema `active-perception-v0.1`, and an exact member target ID. It validates source identities and nested cue/target contracts before using them.

Consumed plan fields: plan_id; scene_id; timestamp; session_id; coordinate_frame_id; source_topological_state_id; source_complex_state_id; source_physical_state_id; cues and targets for membership/lineage validation; uncertainty. Target fields consumed: target_id, target_scope, object_ids, relation_types, cue_ids, source_ids, rationale_code, priority_band, uncertainty. Source provenance participates in identity validation; it is not reinterpreted as grounding evidence.

Existing scopes remain object, object_set, relation, and subsystem. No target object or relation type is invented. Plan priority is preserved only as source_priority_band metadata.

## 4. External backend boundary

There is no external backend implementation or invocation. `integrate_observations(request, observations, *, response_status='completed')` consumes an explicit tuple/list of typed observations. The caller declares completed, unavailable, or indeterminate status. None is not a no-candidate report and is rejected.

No model, image reader, network client, repository integration, filesystem cache, or inference framework exists in the adapter. Tests supply deterministic synthetic data only.

## 5. GroundingFrame

Fields: image_id, width, height, timestamp, session_id, coordinate_frame_id, scene_id, provenance, frame_id, schema_version. Width/height are positive integers, excluding booleans; timestamp is finite and nonnegative; IDs are nonempty. The content-derived frame_id binds dimensions, image identity, temporal context, and metadata.

This is metadata only. Raw bytes, image objects, tensors, and buffer fields are rejected by structured helpers. Dimensions are never inferred from returned coordinates.

## 6. GroundingRequest

Fields: source_plan_id, source_target_id, target_scope, object_ids, relation_types, symbolic_query, frame, requested_output_types, uncertainty, provenance, request_id, schema_version.

The symbolic query contains exactly target_scope, object_ids, relation_types, and semantic_query_status='unavailable'. The core does not turn o1/o2 into natural-language descriptions. Caller phrase mappings and semantic resolvers are deferred. Missing labels remain unavailable; no class similarity or unsupported relational phrase fills the gap.

Only box and point are implemented. Region-reference and segmentation-mask output types are explicitly rejected in v0.1. Output type order and duplicates canonicalize to a sorted set.

## 7. RawGroundingObservation

Fields: request_id, frame_id, backend_name, backend_version, output_type, coordinates, coordinate_space, optional backend_label, optional backend_score, optional raw_reference, uncertainty, provenance, observation_id, schema_version.

Frame ID is mandatory in addition to request ID. Backend name/version are explicit nonempty strings (a caller may explicitly use an unknown-version label). Optional label/reference must be nonempty strings when supplied. Coordinates must have exact box/point arity, finite numeric values excluding booleans, nonnegative values, and non-reversed box ordering. Normalized bounds are checked at construction; pixel upper bounds require frame dimensions and are checked during integration. Backend scores must be finite if supplied, but no [0,1] assumption is made. Opaque backend objects and raw payloads are rejected.

## 8. Coordinate spaces

Supported spaces: normalized_0_1000, normalized_0_1, pixel.

| Source | Pixel conversion |
|---|---|
| normalized_0_1000 | x / 1000 * width; y / 1000 * height |
| normalized_0_1 | x * width; y * height |
| pixel | Preserve after bounds validation |

Coordinates use continuous image-edge semantics: x is in [0,width] and y in [0,height], including endpoints. Thus 1000 maps to width/height, not width-1/height-1. This also applies to points; they are image-space locations, not integer array indices. Zero-area boxes are allowed because x1 <= x2 and y1 <= y2 are the contract.

No integer rounding, clipping, clamping, reversed-box repair, or nearest-frame matching occurs. Floating-point arithmetic follows the stated operation order. Canonical normalized_0_1 coordinates are calculated from validated pixel coordinates by dividing x by width and y by height. They may retain ordinary floating-point representation effects; no tolerance-based identity matching is performed.

## 9. GroundedRegion

Fields: output_type, pixel_coordinates, normalized_coordinates, frame_width, frame_height, source_observation_id, frame_id, uncertainty, provenance, region_id, schema_version.

Regions validate both coordinate representations and their consistency, dimensions, ordering, and bounds. Provenance retains source_coordinate_space, source_coordinates, and the conversion rule. A region is a 2D localization candidate only; it supplies no depth, 3D center, world coordinate, physical object, or track association.

## 10. GroundedObservationTarget

A grounded observation target is a localization candidate attached to a symbolic observation target; it is not a verified physical fact.

Fields: source_plan_id, source_target_id, source_request_id, target_scope, object_ids, relation_types, frame, regions, grounding_status, backend_metadata, uncertainty, provenance, grounded_target_id, schema_version.

Exactly one region yields grounded_candidate; more than one yields multiple_candidates; a completed empty response yields no_candidate. Explicit unavailable/indeterminate responses yield the corresponding status and must contain no regions. Malformed or incompatible observations are rejected, rather than converted into a partial success. Region frame and backend metadata observation lineage must agree.

## 11. Multiple candidates

Multiple grounding candidates are preserved; Step 26B does not choose a winner.

All distinct valid observations become regions, including identical geometry with different backend records, labels, scores, or references. Exact duplicate observation IDs coalesce. Multiple candidates add multiple_candidate_ambiguity. They do not establish multiple real-world objects. A malformed observation in a mixed response rejects the whole response; valid entries are never silently salvaged.

## 12. No-candidate semantics

No grounding candidate does not imply physical absence.

An explicitly supplied empty completed sequence reports no candidates under that backend response. No observations are synthesized. An unavailable backend is not a no-match report. Completed response provenance marks external_grounding_evidence=True, including an explicit empty result; unavailable/indeterminate responses mark it False. Neither is negative physical evidence.

## 13. Backend score semantics

A backend score is backend metadata, not AISTHESIS confidence, Bayesian posterior, risk, or truth probability.

Scores and labels are preserved in backend_metadata linked by source_observation_id. They are not normalized, calibrated, fused, thresholded, or used to select/rank regions. Step 26A priority does not affect score interpretation, and backend scores do not change that priority. Scores can affect content-derived IDs because they are preserved evidence metadata; canonical identity order is not a score ranking.

## 14. Uncertainty

Requests union plan uncertainty, target uncertainty, core limitations, and semantic_query_unavailable. Regions union request uncertainty, backend record uncertainty, and coordinate limitations. Integrated targets union request/region/backend uncertainty and add multiple_candidate_ambiguity or backend_response_unavailable/backend_response_indeterminate where applicable.

Core limitations: grounding_not_physical_truth; pixel_location_not_world_location; backend_label_not_verified_identity; backend_score_not_aisthesis_confidence; no_candidate_not_physical_absence; grounding_not_control; external_backend_not_validated_by_step26b; continuous_image_edge_coordinates. No limitation becomes numeric confidence.

## 15. Provenance

The policy explicitly states:

```text
advisory_only = True
executable_control = False
physical_truth_claim = False
hidden_actor_inference = False
object_identity_verified = False
Bayesian_feedback = False
ExperienceLearning_feedback = False
branch_selection = False
risk_update = False
world_state_mutation = False
backend_score_promoted = False
task_level_validation = False
```

Requests additionally record external_grounding_evidence=False; source topological/complex/physical IDs; source cue IDs and source IDs; source priority and rationale; separate source plan/target uncertainty; semantic query unavailability and deferred resolution. Integrated targets retain these and add source_frame_id, source_image_id, response_status, winner_selection=False, duplicate_policy, and external_grounding_evidence according to response status.

Backend metadata separately retains backend name/version, label, score, raw reference, uncertainty, and frozen backend provenance. Caller/backend metadata is not promoted into the core policy. Regions retain original coordinate values and conversion provenance. Missing physical IDs remain None rather than being invented.

## 16. Alignment

Frame and plan must exactly match scene, timestamp, session, and coordinate frame. Source and nested contract versions/content-derived identities are checked. Target ID must belong to the supplied plan. Raw observations must match the exact request_id and frame_id and use a requested output type. A different image with otherwise identical metadata still creates a different frame/request identity and cannot reuse results. Grounded regions must match the result's frame ID and dimensions.

No timestamp tolerance, coordinate-frame transformation, scene reconciliation, or identity matching is performed. The complete lineage is grounded target -> request -> Step26A target/plan -> source topological state, using exposed IDs only.

## 17. Determinism

Frozen dataclasses, frozen nested mappings, detached bounded serialization, sorted symbolic sets, canonical region/backend metadata ordering, and existing stable_id provide deterministic JSON and IDs. Observation permutations produce identical canonical output. Identical observations deduplicate by their content-derived IDs. No random value, UUID, wall clock, model state, or external state participates.

## 18. Why grounding != truth

Grounding does not imply a hidden actor.

A result records externally supplied localization evidence. It does not verify the world, a causal link, a latent physical object, or a hidden actor. The core does not infer additional objects from uncertainty or labels.

## 19. Why grounding != identity

A bounding box does not verify object identity.

A backend label is not a verified semantic identity.

Symbolic object IDs remain target references, not backend-verified track IDs or identity assignments. A relation target with multiple IDs does not gain per-object associations from one returned box.

## 20. Why pixel location != physical location

A pixel location is not a metric physical-world location.

A point is not a 3D object center. No camera calibration, depth, sensor fusion, metric conversion, physical distance, or world-coordinate inference is implemented.

## 21. Why grounding != control

Grounding does not issue camera, robot, gaze, navigation, or actuator commands.

The module has no action policy, sensor movement, trajectory, pan/tilt/zoom command, or robot pose output. Grounding is localization evidence only.

## 22. LocateAnything compatibility

The supplied Step 26B brief describes NVIDIA LocateAnything as supporting generalist visual-language grounding/detection/pointing and normalized [0,1000] coordinate tokens that can be parsed into boxes/points. This implementation uses that supplied compatibility description; no external repository, worker, release, or live backend was inspected or tested.

Compatibility here means the core can accept box/point observations expressed in normalized_0_1000, normalized_0_1, or pixel space. It is not an integration or verification of NVIDIA's parser/API. No NVIDIA code is copied and no Eagle, transformers, or model package is required.

## 23. Future real backend integration

A future external adapter could expose this conceptual interface:

```python
class LocateAnythingBackend:
    def ground(self, image, request) -> tuple[RawGroundingObservation, ...]:
        ...
```

This is documentation only. Such an adapter would own model loading, semantic query resolution, token parsing, image handling, and exact request/frame association outside the core. The AISTHESIS adapter depends only on its own typed raw-observation contract. Real inference and backend behavior need separate validation.

## 24. Relationship to Step27

Calibration and broader uncertainty remain future work. Backend scores do not become calibrated confidence here.

Step 26B does not update Bayesian beliefs.

Step 26B does not modify ExperienceLearning.

Step 26B does not select future branches.

Step 26B does not mutate the world model.

## 25. Relationship to Experiment001

No improved prediction, hidden-actor anticipation, real visual-grounding benefit, active camera control, or Experiment001 success is established. Future experiments may evaluate whether localization evidence helps inference, with appropriate ablations.

## 26. Explicit non-goals

No external backend, model download, inference, network call, repository clone, dependency installation, mask grounding, region-reference resolution, caller-label mapping, sensor reading, control, Bayesian/ExperienceLearning feedback, branch choice, risk update, world-model mutation, production wiring, Mini-Lab edit, or paper-input edit. Exactly the new module, dedicated tests, and this document are created. Existing files and old untracked artifacts are untouched. Nothing is staged, committed, pushed, or tagged.

## 27. Validation status

Dedicated synthetic suite: **130 passed**. It covers frozen contracts, deterministic IDs/JSON, detached serialization, non-mutation, ordering, frame dimensions/time/alignment, source lineage, symbolic queries, coordinate spaces/conversion/bounds/arity, malformed mixed results, duplicate handling, score metadata, all result statuses, and no network/filesystem calls during integration.

Requested regressions: **1,112 passed** (23.69 seconds). Step 26A: 74; Step 25: 69;
Step 24: 55; Step 23: 87; Step 22: 80; Step 21: 62; Step 20: 103;
Step 19: 117; Step 18: 144; Step 17: 93; world-model suites: 228
(physical world 31, physics constraints 45, latent physical state 52,
hybrid world state 39, common evidence state 61). Including dedicated tests,
**1,242 tests passed**.

Final `git diff --stat` is empty because the three additions remain untracked.
`git status --short` shows these three files beyond the pre-existing untracked
artifacts. No tracked production file changed and nothing was staged.

Step 26B is an engineering grounding-interface layer, not evidence that spatial grounding improves AISTHESIS.

- IMPLEMENTED
- TESTED (synthetic engineering contracts)
- NOT YET TASK-LEVEL EVALUATED
- NOT YET EXPERIMENTALLY SUPPORTED
