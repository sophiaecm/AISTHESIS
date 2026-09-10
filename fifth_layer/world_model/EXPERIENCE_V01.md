# Prediction error + experience v0.1

Implemented locally on `d0e3840` (`sensory-evidence-integration-v1`). This is an
observational recorder and comparator. It performs no learning, calibration,
parameter updates, hypothesis revisions, or inference feedback.

## Inspection and reuse

Inspected before editing: repository structure; world-model `experience_memory.py`,
`trajectory.py`, `hypothesis.py`, `hypothesis_generator.py`, `scene_state.py`,
`adapter.py`, `evidence.py`, `evidence_providers.py`, `sensory_evidence.py`,
`_structured.py`, `__init__.py`; legacy `fifth_layer/prediction_error.py`,
`prediction_evaluation.py`, `engine.py`; structured motion/occlusion extraction
and tracking identity fields; existing sensory postprocessing and applicable tests.
Repository-wide searches located existing prediction-error, evaluation, episode,
trajectory and outcome code. No AGENTS.md was found.

Reusable foundations:

- `ExperienceEpisode` already provides prediction/observation/error summary slots,
  scene/hypothesis/trajectory/prediction links, timestamps, status and provenance.
  Its fields and validation behavior were not changed.
- `ExperienceMemory` already supports a maximum of 512 entries, a 60-second TTL
  after admission, oldest-insertion eviction, duplicate episode/prediction ID
  rejection, injectable clock, and `RLock` protection. Its implementation is unchanged.
- `FutureTrajectory` supplies explicitly timed state points and optional centers,
  boxes, uncertainty, hypothesis/track linkage and provenance. Existing
  `adapt_temporal_trajectories` projects previously produced temporal paths.
- `Hypothesis` supplies source/target time, horizon, track ID, evidence references,
  scores, confidence, assumptions and provenance. Object association can appear
  in its v0.2 provenance. These contracts and generation rules are unchanged.
- `SceneState` separates observed objects from predicted tracks. The existing
  adapter excludes predicted/non-observed detections. Motion extraction supplies
  `motion_state`, track identity and observed centers. Occlusion extraction measures
  box overlap/truncation only: it explicitly does not confirm true occlusion.
- Existing `PredictionError(details=...)` is reused by
  `PredictionEvaluation.as_prediction_error()`. The older live evaluator also
  contains calibration coupling, thresholds and UUIDs; it is left untouched and
  is not called by this observational loop.

## Files changed

Modified only:

- `fifth_layer/world_model/experience_memory.py`: add `unobservable` and
  `insufficient_evidence` to `EvaluationStatus`, retaining every existing value.
- `fifth_layer/world_model/__init__.py`: export the new records and functions.

Created:

- `fifth_layer/world_model/prediction_records.py`
- `fifth_layer/world_model/prediction_observation.py`
- `evaluation/experience.py`
- `test_prediction_experience.py`
- `evaluation/tests/test_experience.py`
- this report, `fifth_layer/world_model/EXPERIENCE_V01.md`

## Exact records

All three new records are frozen dataclasses with detached immutable mappings.
IDs from the projection/evaluation functions use the existing canonical SHA-256
`stable_id` utility; callers constructing records directly supply their own IDs.

| Record | Fields |
| --- | --- |
| `PredictionRecord` | `prediction_id`, `scene_id`, `hypothesis_id`, `hypothesis_type`, `source_timestamp`, `target_timestamp`, `horizon_seconds`, `predicted_state`, `track_id`, `object_id`, `trajectory_id`, `image_size`, `confidence`, `uncertainty`, `evidence_for`, `evidence_against`, `provenance` |
| `OutcomeRecord` | `outcome_id`, `prediction_id`, `scene_id`, `observation_timestamp`, `association_status`, `observed_state`, `track_id`, `object_id`, `image_size`, `association_confidence`, `unavailable`, `provenance` |
| `PredictionEvaluation` | `evaluation_id`, `prediction_id`, `outcome_id`, `status`, `evaluated_timestamp`, `metrics`, `reasons`, `missing_fields`, `provenance` |

Prediction time must satisfy source + horizon = target. Optional identity,
geometry, confidence, uncertainty, trajectory and observation time remain null
when unavailable. Unassessed v0.2 confidence is represented as unknown; heuristic
prior/posterior values remain in source provenance and are not treated as accuracy.

`predicted_state` can retain `event`, `motion_state`, `visibility`, `center`,
`bbox`, and explicitly supplied `event_timestamp`, `modality`, `sensory_claim`.
The hypothesis projection copies only an explicitly timed trajectory point at
the hypothesis target. Untimed arrays are not assigned invented timestamps.
Boxes are retained when present; no boxes or centers are synthesized in forecasts.
Only center displacement is evaluated in v0.1, not box overlap error.

`observed_state` contains independently available later `motion_state`, `center`,
`visibility`, explicit `events={event_name: bool}`, optional `event_timestamp`
and observed `sensory_claim`. Observed centers may be derived arithmetically from
actual detector boxes in existing xyxy or xywh format. Missing boxes stay missing.
Predicted/inferred outcome flags are rejected. Unmatched records cannot assert
object observations. Raw source evidence and source paths remain in provenance.

The episode factory uses existing `prediction_summary`, `observation_summary`,
and `prediction_error` for the three detached records, retaining ID/time links.
No second episode or memory architecture was introduced. `as_prediction_error()`
offers the existing legacy error wrapper without invoking legacy comparison logic.

## Observation and association policy

`outcome_from_scene` matches exactly typed track IDs within a continuous tracking
session. Integer `7` and string `"7"` are different identities. It does not rematch
by class, proximity, semantics or the predicted location. Duplicate observed IDs
and explicit `association_reliable=False` are ambiguous. Supplied association
confidence is retained, including for unreliable matches; detector confidence
is never substituted and a perfect association score is never manufactured.

Legacy `object_id` and `current_object_id` can be frame-local indices, so they
cannot establish identity across frames by default. The direct API accepts
`object_ids_persistent=True` only for callers with an explicit persistent-ID
contract. The saved-result utility does not enable that option. The utility
separates modes and source-video hashes; the direct sequence API requires a
single continuous tracking session and camera coordinate system.

Outcome sources are `SceneState.observed_objects`, matching fresh motion evidence,
and explicitly labelled later observation annotations. The latter may be supplied
in `scene.occlusion_evidence['occlusion_evidence']` with `track_id`,
`epistemic_status='observed'`, and `visibility='occluded'` or `'visible'` (plus
source provenance). This is an optional explicit annotation contract, not a claim
that existing box-overlap extraction emits true visibility measurements.
An observed object may also carry explicit visibility/event annotations.

The adapter ignores predicted tracks, inferred visibility, stale timestamps,
semantic descriptions/omissions and inferred sensory consequences. A detector
box alone is not evidence of complete visibility: partial occlusion can coexist
with detection. Raw overlap or frame truncation is not confirmed occlusion.
Absence of a track neither confirms occlusion/disappearance nor contradicts it.
Conflicting later values become unknown with a recorded conflict reason.

Sensory comparison requires explicitly `observed` sensory EvidenceItems with
matching modality, identity, scene and time. `expected`, `inferred`, and
`unavailable` cannot satisfy a measurement requirement. A standalone auditory
expectation has no native target horizon; the saved-result utility therefore
does not invent a forecast horizon for it. It evaluates issued world-model
hypotheses; explicitly timed sensory PredictionRecords are supported by the API.

## Evaluation and metrics

Statuses reuse `EvaluationStatus`:

- `supported`: all requested comparable components agree, with none missing.
- `contradicted`: observed components disagree and none agree; missing components
  do not contribute negative evidence.
- `partially_supported`: some components agree while others disagree or remain
  unavailable.
- `unobservable`: no associated later observation, or a requested sensory outcome
  has no measurement.
- `insufficient_evidence`: unknown/ambiguous identity, unsuitable time, or no
  comparable supported fields. This is not a prediction failure.

Existing `pending`, `unevaluable` and `expired` values remain valid for existing
callers. The new evaluator never changes a Hypothesis status or score.

Metrics always retain unknown values as null:

| Metric | Conditions |
| --- | --- |
| `event_match` | Explicit later event boolean; for existing motion/visibility hypothesis types, compatible target motion/visibility can supply the state-level comparison. Other event occurrences are not negative evidence. |
| `motion_state_match` | Both motion labels available. Directional predictions require directional evidence; a generic later `moving` does not disprove a specific direction. |
| `visibility_match` | Explicit later visible/occluded state. |
| `position_error_pixels` | Euclidean center displacement, both centers present and known equal image dimensions. |
| `normalized_position_error` | Pixel error divided by the common image diagonal. |
| `position_match` | Error within explicitly recorded positional tolerance. |
| `temporal_error_seconds` | Signed observed minus predicted event time, only with explicit matching event occurrence and both event times. Snapshot sampling lag is not event-timing error. |
| `sensory_match` | Explicit observed sensory claim compared with an explicitly supplied expected claim. |
| `association_confidence` | Source-supplied association confidence, otherwise null. |

Default comparison policy uses exact target time and exact center equality
(tolerances 0). Optional nonnegative time/position tolerances are recorded in
evaluation provenance; they affect comparison only, never inference thresholds.
The evaluator requires a later observation and does not silently compare a
one-second forecast with a half-second snapshot. Changed/unknown geometry yields
unknown position metrics, never a zero error. Actual identical centers yield zero.

Evaluation is target-snapshot compatibility under partial observability. Existing
hypotheses say what *may* happen; a contradicted target snapshot does not prove
the original possibility impossible, establish event history, or calibrate its
heuristic score. Reappearance/occlusion transition timing is not reconstructed
from sparse snapshots.

## Determinism, recording and manual utility

`prediction_from_hypothesis`, `outcome_from_scene`, `evaluate_prediction`, and
`experience_episode` are pure projections/comparisons. They use explicit source
times and canonical IDs, not wall-clock time or randomness. Evidence references,
source records, conflicts and diagnostic lists are canonicalized where unordered.
Input frame order does not affect sequence evaluation. Duplicate frame timestamps
or scene IDs in one session are rejected rather than arbitrarily selected.

`evaluation.experience.evaluate_sequence` takes structured scenes with their
already-issued HypothesisSets and EvidenceBundles (and optional trajectories).
For each forecast it selects the closest later snapshot within target-time
tolerance, with an earlier-time tie break. It never chooses observations by
evaluation quality. Every alternative and final-frame forecast is retained in
the audit report, including those without later evidence.

Episodes are inserted into the existing ExperienceMemory in deterministic replay
time order using its existing injectable clock. Capacity, duplicate protection,
TTL, eviction and locking stay unchanged. The report lists all evaluations plus
the bounded memory's retained IDs and summary. This report export is not a new
persistent memory backend. No memory read affects inference.

After review, examples for saved combined sensory results:

```powershell
.venv/Scripts/python.exe -m evaluation.experience results_case_a_combined_sensory.json --output results_case_a_experience.json
.venv/Scripts/python.exe -m evaluation.experience results_case_b_combined_sensory.json --output results_case_b_experience.json
```

The Fifth Layer-only sensory files can be used with distinct output names such
as `results_case_a_fifth_layer_experience.json`. This utility reads existing JSON
only. No video decoding, VLM/detector call, reasoner call, or forecast regeneration
occurs. Frames lacking saved `sensory_world_model` hypotheses are explicitly
reported as skipped, not assigned retrospective forecasts. Rejected and VLM-only
frames cannot become physical outcome evidence. Source JSONs are not rewritten;
output must be a new `*_experience.json` path and is opened exclusively. The source
file hash and comparison policies are recorded. No real result postprocessing or
real-video/VLM evaluation was run during implementation.

## Validation and safety report

**377 unit/regression tests passed: 311 existing + 66 new.**

| Suite | Tests |
| --- | ---: |
| Existing World Model v0.1 | 66 |
| Existing World Model v0.2 | 46 |
| Existing sensory evidence | 34 |
| Existing evaluation harness | 24 |
| Existing sensory postprocessing | 5 |
| Other existing regression tests | 136 |
| New prediction/experience contracts, comparisons and memory tests | 52 |
| New offline experience utility tests | 14 |
| Total | 377 |

The new tests cover construction, timestamps/horizons, provenance, stable IDs,
geometry and null metrics, all five evaluation states, motion/visibility/event
comparison, timing, track loss, detector absence, semantic omission, sensory
absence versus observation, ambiguous/persistent identity, permutations, episode
links, duplicates/capacity/TTL, unchanged hypotheses and sensory evidence after
recording, no model/reasoner/calibration calls, frame selection, mode isolation,
JSON round trips and refusal to overwrite existing results. Existing tests were
not weakened or changed.

Regression command used:

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$testModules = Get-ChildItem -Filter 'test_*.py' | Where-Object { (Get-Content -LiteralPath $_.FullName -Raw) -match 'unittest|def test_' } | ForEach-Object { $_.BaseName }
.venv/Scripts/python.exe -m unittest @testModules evaluation.tests.test_harness evaluation.tests.test_sensory evaluation.tests.test_experience -q
```

Eight deterministic legacy smoke scripts also exited successfully:
`test_active_perception.py`, `test_auditory.py`, `test_auditory_negative.py`,
`test_occlusion_reasoner.py`, `test_orchestrator.py`, `test_semantic_conflict.py`,
`test_temporal.py`, `test_yolo_occlusion.py`. These are not additional unit-test
assertion counts. Five model/service smoke scripts remain intentionally excluded:
`test_active_perception_loop.py`, `test_locate_adapter.py`, `test_perception_fusion.py`,
`test_smolvlm_core.py`, `test_yolo_detector.py`. They launch neural inference or
external services at import time, outside the authorized deterministic validation.

`git diff --check` passed. Only the two existing files listed above are modified;
six implementation/test/documentation files are new. The ten baseline/sensory
JSONs and `test_videos/` were already untracked and remain untracked. HEAD remains
`d0e3840`. No commit, push or tag operation occurred.

Confirmed untouched: hypothesis generator and scores; sensory inference and
semantics; `live_app.py`; SmolVLM; detectors/tracking; all reasoners and the
orchestrator; existing evaluation harness/live evaluator/calibration code; all
existing tests. ExperienceMemory has no feedback path to those components.
No learning, training, weight updates, downloads, active-perception implementation,
risk changes, or external world models were introduced.

All ten result JSONs have identical SHA-256 values before and after implementation:

| File | SHA-256 |
| --- | --- |
| results_case_a_combined.json | A2C478A99C107C95EC07975517F8DED58963C8CDDBEE0052166B01C29141DE7D |
| results_case_a_fifth_layer.json | 1D0866C30DC4E98B2CC17DB25EE40D7829C4354058D7D87886B693D1C15D0759 |
| results_case_a_vlm.json | 8C137B8E1F61C1F855F4FA0B33CFE2670620C35C323C60CA89DDDDB62EDA3F9E |
| results_case_b_combined.json | 955F4886D11987B08D06742038676C3B70705A256DF7C17ADABCAC62E6267B06 |
| results_case_b_fifth_layer.json | EA8FFC93F80C1EEA4C18FC78A0EE6BFF2AFE39DC6B4FB78B68AE552D14891C0E |
| results_case_b_vlm.json | EE8882BF018788613E185F902D451EFCBA5A4AB704AF2E0133259EFC3297E000 |
| results_case_a_combined_sensory.json | 12DF861878B6EE941E60360B4A89042065D564B20085AD96520ABC876CE2876B |
| results_case_a_fifth_layer_sensory.json | F017D2FAAC4EB4EC7D59A9777B3F7E3A2FD79729B2B3FDBE92193B97598319C1 |
| results_case_b_combined_sensory.json | 0BB3B84AD83BF3A232477C185C455C4113062448668A5AF495087A78A33DF3D5 |
| results_case_b_fifth_layer_sensory.json | B0EF065DBCD14CF232A528725801BB87946CB7B01DBB1AC43A2BDCA0E21DBC54 |
