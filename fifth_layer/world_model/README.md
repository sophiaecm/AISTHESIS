# World model v0.1 foundation

The opt-in v0.2 evidence providers and multi-hypothesis generator are documented
in [V02.md](V02.md). The v0.1 contracts below retain their existing behavior.

This opt-in package defines data contracts only. Existing WorldState, inference,
reasoners, live prediction selection, evaluation and calibration remain unchanged.
No module in the existing pipeline imports this package. Importing it does not
create memory, generate hypotheses, evaluate predictions or invoke a model.

## Contracts

- `SceneState`: detached structured scene, with observed objects explicitly
  separate from predicted tracks. Unknown timestamp, geometry, source ID and
  snapshot sequence remain `None`. `semantic_evidence` is an additional field
  preserving the repository's semantic outputs without calling semantic reasoning.
- `Hypothesis`: immutable caller-supplied claim, probabilities, evidence, temporal
  metadata, status and provenance. Track IDs retain integer/string representation.
- `HypothesisSet`: one scene's ordered hypotheses; duplicate IDs and mismatched
  scene IDs are rejected. `normalized_posteriors` is a read-only ID-to-probability
  mapping for **active** entries only. It divides by their posterior sum, or uses
  `1/N` when that sum is zero. Other statuses remain in `hypotheses` unchanged.
  Empty/no-active sets return an empty mapping. Original hypotheses, prior and
  raw posterior values are never rewritten. `highest_probability_hypothesis`
  considers active hypotheses, breaks ties by input order, and returns `None`
  when none are active; it does not select anything in the running application.
- `FutureTrajectory`: existing `(x, y)` centers and `(x1, y1, x2, y2)` bboxes,
  optional structured predicted states and per-step uncertainty. Nonempty step
  arrays must have matching lengths; absent arrays are allowed. Geometry may
  lie outside the image. `validate_trajectories` checks duplicate trajectory IDs
  and references against a supplied HypothesisSet, returning a tuple. There is
  no registry or trajectory generator.
- `ExperienceEpisode`: links source/target scenes, hypothesis, trajectory and
  prediction IDs with detached hypothesis/prediction/observation summaries,
  timestamps and a caller-supplied evaluation status. A Hypothesis instance can
  be supplied as the snapshot; its fields become an immutable mapping. Snapshot
  IDs, when present, must match episode links. Optional `prediction_error` holds
  externally supplied structured data (including existing PredictionError
  dataclasses); this package defines no metric and calculates no error. Missing
  observations do not change status. `unevaluable` and `prediction_error=None`
  are valid for insufficient evidence.
- `ExperienceMemory`: recorder with `add`, `get`, `recent`, `cleanup`, `summary`.
  Admission TTL is exactly 60 seconds using `time.monotonic`; tests inject a
  clock. Capacity defaults to 512 and may be lowered, never increased. Full
  memory evicts the oldest insertion. Duplicate episode IDs and non-null
  prediction IDs raise ValueError among retained entries, before eviction.
  Expiry/eviction releases both IDs. Reads do not refresh TTL/order. All methods
  serialize through one RLock and clean expired entries. `recent` returns an
  immutable tuple, newest insertion first. Summary is a detached count/status
  dictionary. There is no persistence or feedback into inference.

## WorldStateAdapter

Use `WorldStateAdapter.from_world_state(state, scene_id=..., provenance=...)`.
An omitted scene ID is a new UUID, not a fabricated WorldState identity. The
legacy WorldState has no native ID or sequence. Callers may supply
`source_world_state_id` and `snapshot_sequence_id` from their own metadata;
the latter matches the name exposed by DeepAnalysisState job metadata.

All fields below are read from `WorldState.data` except `WorldState.timestamp`.
Derived reasoner fields are copied only when already present. This adapter does
not fetch orchestrator results or run reasoners to populate missing evidence.

| Source | SceneState destination |
| --- | --- |
| `WorldState.timestamp` | `timestamp` and provenance source timestamps |
| `image_width`, `image_height` | same names |
| `accepted_detections` (if present), otherwise `detections` | `observed_objects` |
| `predicted_tracks` | `predicted_tracks` |
| `scene_relations` | `spatial_relations` |
| `motion_evidence` | `motion_evidence` |
| `occlusion_evidence`, `occlusion_reasoning` | same-named entries in `occlusion_evidence` |
| `position`, `velocity`, `dt`, `occlusion_zone`, `physics_hidden_interaction_possible`, `physics_confidence` | same-named entries in `physics_evidence` |
| `latent_occlusion_state`, `latent_temporal_state`, `latent_scene_state`, `latent_hypothesis`, `latent_physical_risk`, `latent_risk`, `auditory_latent_state` | same-named entries in `latent_evidence` |
| `semantic_evidence`, `scene_description` | same-named entries in `semantic_evidence` |
| `risk_level` | `risk` |
| `uncertainty`, otherwise `fused_uncertainty` | `uncertainty` |
| `observation_timestamp`, `snapshot_timestamp`, `fast_scene_timestamp` | provenance source timestamps |
| `source_type`, `source_path`, `model`, `perception_sources` | provenance source metadata |

Empty accepted detections never fall back to rejected detections. Explicit
predicted or non-observed records in observation input are excluded, never moved
into predicted tracks. Legacy unlabelled perception records remain observations.
Tracking IDs, timestamps and pixel uncertainty remain inside their summaries;
`position_uncertainty` is not reinterpreted as a probability. Risk is a supplied
label, not a newly inferred score.

Provenance records the actual source field mapping, source timestamps, known
perception metadata, adapter/source component names, schema `0.1`, optional caller
IDs and a separately namespaced `supplied` provenance mapping. Known caller
software versions can be retained there; no upstream software version is invented.

## Immutable summary boundary

Frozen dataclasses use detached tuples, recursively copied read-only mapping
proxies and frozensets. Dataclass payloads become structured mappings. No mutable
lists/dicts/sets or opaque object references are retained. Direct construction
rejects bytes, memoryviews, ndarrays, PIL objects, tensors, unsupported objects,
buffer-named fields and strings over 16,384 characters. Adapter conversion drops
unsupported leaves and buffer-named entries instead of copying them; unselected
WorldState fields are never traversed. This applies equally to provenance.
Snapshot `pixels`, `state_json`, `motion_json` and frame buffers are not copied;
pass `snapshot.world_state()` to adapt its structured state. The adapter does
not accept the image-bearing AnalysisSnapshot itself.

Validation rejects non-finite/out-of-range probabilities and confidence,
normalized uncertainty, negative timestamps/horizons, malformed centers/bboxes,
invalid statuses and inconsistent IDs. Legacy `box` uses xywh, while
`box_xyxy`/`bbox` use xyxy. Zero-sized boxes and unknown optional metadata remain
valid. Cyclic summary structures produce descriptive errors.

## Validation commands

```powershell
.venv/Scripts/python.exe -m unittest test_world_model -q
.venv/Scripts/python.exe -m unittest test_track_memory test_tracking_feedback test_predicted_tracks test_observation_lifetime test_prediction_evaluation test_confidence_calibration test_analysis_connections test_deep_analysis test_live_identity test_live_integration test_fast_scene_connection test_fast_scene_stability test_scene_lifetime test_structured_scene_narrator -q
```

No model-loading scripts or camera tests are required for this data-only package.
Future v0.2 can introduce explicit evidence-provider adapters around existing
Physics/Temporal/Occlusion/Semantic reasoners and then a separate multi-hypothesis
generation stage. None of that behavior is implemented here.
