# Step 21 — ASTRA–EFA Experience Integration v0.1

## 1. Purpose

Represent internal longitudinal change across completed evaluated encounters.
This is opt-in engineering representation, with no application wiring, inference,
prediction generation, storage service, reward or training. Only three Step 21
files are added; existing world-model contracts remain frozen.

## 2. ASTRA–EFA conceptual relationship

`AstraSelfState`, `EncounterEncoder`, `TransformationEncoder`,
`AstraExperienceState`, and `RecursiveUpdater` implement conservative structural
analogues of the requested concepts. This is neither biological equivalence nor
scientific validation. No external ASTRA–EFA implementation is imported.

## 3. Existing ExperienceEpisode relationship

`EncounterEncoder.encode(episode)` accepts an existing `ExperienceEpisode`.
The episode is detached and reconstructed through its existing constructor.
Prediction, outcome and evaluation summaries must contain exactly all fields of
their existing dataclasses, including optional fields explicitly present as
`None` or empty structures. Their existing constructors validate the content.
Missing fields are rejected, not filled from defaults. No evaluation is rerun.
Source IDs are retained as attributed IDs, not asserted to be content hashes.

## 4. Raw vs experience history

`trajectory.raw_episode_ids` / `raw_history` are ordered source episode IDs.
The corresponding full detached episode remains in each encounter's
`source_episode`; encounter IDs are available in the same order.
`experience_ids` contains ordered experience IDs. `experience_history` is an
ordered tuple of `(experience_id, transformation_id)` pairs. These are separate
representations; no source episode is overwritten with a transformation.

## 5. SelfState semantics

`AstraSelfState` is a frozen dataclass with these exact fields:

| Fields | Definition |
| --- | --- |
| `state_id`, `schema_version` | Content-derived ID and `astra-efa-experience-v0.1` |
| `session_id`, `timestamp` | Explicit session; initial time or latest evaluation time |
| `sequence_index` | Number of completed experiences |
| `completed_experience_ids`, `completed_episode_ids`, `completed_prediction_ids` | Ordered, unique, aligned histories |
| `evaluation_statuses` | Ordered terminal statuses aligned with those histories |
| `evaluation_counts` | Sorted status-to-count mapping, including zeros |
| `cumulative_comparable_predictions` | Supported + partially supported + contradicted counts |
| `cumulative_supported`, `cumulative_partially_supported`, `cumulative_contradicted`, `cumulative_unobservable`, `cumulative_insufficient_evidence` | Individual terminal-status counts |
| `cumulative_error_summary` | Empty mapping; aggregation deferred |
| `last_experience_id`, `previous_state_id` | Last encounter's experience and predecessor state; initially `None` |
| `provenance` | Explicit representation-only policy |

`initialize(session_id=..., timestamp=...)` returns sequence zero, empty
histories, zero counters and no fabricated metrics. It is a neutral software
initialization, not a biological baseline or physical world state.

## 6. Encounter encoding

`AstraEncounter` has exact fields: `source_episode`, `encounter_id`, `session_id`,
`source_episode_id`, `prediction_id`, `hypothesis_id`, `source_scene_id`,
`target_scene_id`, `prediction_timestamp`, `observation_timestamp`,
`evaluated_timestamp`, `evaluation_status`, `prediction_summary`,
`observation_summary`, `evaluation_summary`, `source_belief_state_id`,
`source_future_id`, `provenance`, and `schema_version`.

Only `source_episode` is a constructor input; all projections are derived.
Episode, prediction, outcome and evaluation links, scene IDs, hypothesis ID,
trajectory ID and timestamps must agree. Comparable statuses require a matched
association with a non-null, type-exact prediction/outcome identity. Evaluation
status is preserved, not independently declared true by this integration.

## 7. Transformation encoding

`AstraTransformation` has exact fields: `pre_state_id`, `post_state_id`,
`encounter_id`, `sequence_index`, `changed_fields`, `unchanged_fields`,
`delta_summary`, `provenance`, `transformation_id`, and `schema_version`.

`TransformationEncoder.encode(pre, encounter, post)` first verifies that `post`
equals the deterministic recursive update. It compares every serialized
SelfState field, including IDs, timestamps and provenance, in sorted name order.
Each changed field maps to `{before: ..., after: ...}`; unchanged fields are
listed separately. This is a mechanical difference, not improvement or a causal
world explanation. Direct transformation construction validates structure;
use the encoder to verify derivation. ExperienceState always uses the encoder.

## 8. ExperienceState

`AstraExperienceState` has exact fields: `encounter`, `pre_state`, `post_state`,
`transformation`, `experience_id`, `source_episode_id`, `sequence_index`,
`provenance`, and `schema_version`. Only encounter and pre-state are inputs.
The other fields are derived. Its ID uses pre-state ID plus encounter ID before
creating the post-state, avoiding an experience/post-state hash cycle.

## 9. Recursive update

`RecursiveUpdater.update(previous_state, encounter)` returns a new SelfState.
The previous state is untouched; exactly one ID/status is appended to each
history, sequence increases by one, and the predecessor state ID is retained.

| Accepted status | Completed increment | Comparable increment | Status counter increment |
| --- | --- | --- | --- |
| `supported` | 1 | 1 | supported only |
| `partially_supported` | 1 | 1 | partially_supported only |
| `contradicted` | 1 | 1 | contradicted only |
| `unobservable` | 1 | 0 | unobservable only |
| `insufficient_evidence` | 1 | 0 | insufficient_evidence only |

`pending`, `unevaluable` and `expired` are rejected. Missing/unobservable records
never increment the contradicted counter. This follows the existing historical
eligibility terminal set without invoking ExperienceLearning.

## 10. Trajectory

`AstraExperienceTrajectory` fields are `initial_state`, `experiences`,
`trajectory_id`, `session_id`, `current_state`, `raw_episode_ids`,
`experience_ids`, `provenance`, and `schema_version`. It requires a neutral
initial state and validates the complete chain of pre/post states.

```python
from fifth_layer.world_model.astra_efa_experience import AstraEFAExperienceBuilder

builder = AstraEFAExperienceBuilder()
trajectory = builder.build(episodes, session_id="session", timestamp=0.0)
extended = builder.append(trajectory, next_episode)
replayed = builder.replay(episodes, session_id="session", timestamp=0.0)
assert replayed.to_json() == trajectory.to_json()
```

Build/replay consume caller order without sorting or tie-breaking. Append returns
a new trajectory. Earlier experiences and state IDs stay identical. There is no
implicit eviction, TTL, disk persistence or sensor-evidence adapter.

## 11. Prediction-error handling

The complete existing evaluation, including metrics, reasons, missing fields,
evaluation ID and provenance, is preserved without coercing its metric values.
Booleans stay booleans, numeric values retain their value, and `None` stays unknown.
No scalar error, loss function, weighting or calibration is defined.

## 12. Metric aggregation

Deferred in v0.1. `cumulative_error_summary` is always empty, including after
updates. No count/sum/mean is computed. Thus boolean false and missing values
cannot become numeric zero; probabilities and physical errors cannot mix.
Per-episode numeric metrics remain fully available. Pixel errors across frames,
association confidences, and errors under different comparison tolerances need
an explicit comparability contract before longitudinal aggregation is justified.

## 13. Missing/unknown semantics

Missing status, source session, prediction ID, observation timestamp or evaluation
timestamp makes an encounter inadmissible. Missing measurements within otherwise
complete records are preserved. Step 20 can produce an episode from `scene=None`
with no terminal timestamp; such records are deliberately outside this v0.1
longitudinal contract. No timestamp is fabricated to admit them.

## 14. Temporal safety

Prediction source time must strictly precede observation time; observation may
equal evaluation time. Prediction horizon must be positive. Evaluation time must
strictly exceed the prior state timestamp, including the initial timestamp.
Equal timestamps and reversed caller ordering are rejected. Prediction source
times may overlap earlier encounters: issuance can precede another completion.
State time represents completion order, not issuance order.

Declared source timestamps in prediction provenance and hypothesis snapshots
cannot exceed issuance time. Observation state/provenance cannot reference later
source timestamps than the observation; evaluation and episode metadata cannot
reference source times after evaluation. `target_timestamp` is prospective
metadata and is excluded from source-time cutoffs. Expected/inferred/predicted
markers in observed state or explicit observation sources are rejected. Opaque
IDs do not establish undocumented temporal facts. No future episode is consulted
while encoding an earlier encounter.

## 15. Session/identity scope

Prediction provenance must explicitly contain `session_id`; recursively declared
sessions must agree. The updater requires this session to match SelfState.
Duplicate episode or prediction IDs are rejected across the full retained state
history, independent of chronology. Tracker/object IDs are preserved as source
provenance and used only to validate existing within-episode association. They
do not create a persistent identity model or hidden actor.

## 16. Provenance

The detached full episode preserves hypothesis snapshot and recorder metadata.
The complete prediction, outcome and evaluation summaries preserve available
belief IDs, selected hypothesis, future branch, physical sources, evidence
references, issue/outcome/evaluation IDs, schema versions, coordinate frame,
association, assumptions and prior/posterior/likelihood metadata. Those values
are provenance only; they do not affect counters except through the supplied
terminal status, and never supply sensor evidence or Bayesian feedback.
Source records without schema fields do not receive fabricated source versions.

## 17. Determinism

All generated IDs use existing `stable_id`; records use frozen dataclasses,
read-only detached mappings and tuples. Canonical serialization uses sorted keys,
compact JSON and finite numbers. Changed/unchanged field names and status mappings
are sorted. Full source metadata contributes to encounter identity, so changing
provenance changes the encounter ID. Replay under this schema is deterministic;
no wall-clock, randomness, model or external service is used.

## 18. EDIS status

Deferred. No distinctiveness score has a justified comparable-field definition in
this integration. No entropy, surprise, reward, salience or importance is inferred.

## 19. No-learning boundary

No ExperienceMemory admission or mutation, ExperienceLearning call, hypothesis
score change, likelihood generation, Bayesian update, branch selection,
future generation, training, weight adjustment or adaptation claim occurs.
The implementation imports only existing record and structural utility modules.

## 20. Step 22 relationship

Step 22 — Experience Learning v0.2 may later consume auditable trajectories to
test whether historical transformations help inference. That integration and
its empirical evaluation are not implemented here.

## 21. Experiment 001 relationship

Step 21 does not demonstrate better predictions, hidden actor anticipation,
improved TTA or accuracy, calibrated probability, adaptation, learning,
cross-modal benefit, scientific support or Experiment 001 success.

## 22. Known limitations

This is descriptive bookkeeping over attributed evaluations, not verification of
the evaluator's scientific validity or evidence authenticity. No source IDs are
resolved against an external store. Direct SelfState construction validates
internal structure but does not authenticate externally supplied prior history;
use initialization and replay to reconstruct trusted lineage from source episodes.
Missing terminal times and ambiguous ties are rejected. Aggregation and EDIS are
deferred. Histories and embedded pre/post snapshots grow with encounter count;
existing `bounded_plain` limits (50,000 nodes, depth 24, 1 MiB serialization) can
reject large trajectories. This is for small explicit session trajectories, not
an unbounded archive. Repeated prefix construction is not optimized for scale.

## 23. Validation status

- 🧩 IMPLEMENTED
- 🧪 TESTED (synthetic contract and regression tests; results below)
- 📊 NOT YET TASK-LEVEL EVALUATED
- 🏆 NOT YET EXPERIMENTALLY SUPPORTED

Use the existing `.venv/Scripts/python.exe`; the system Python has no pytest.
Dedicated Step 21 tests cover initialization, real Step 20 integration, all five
terminal statuses, incomplete/misaligned records, nested future/session leakage,
expected observations, duplicates, chronology, immutability, exact differences,
metric preservation, independent histories, prefix stability and no feedback.

Final combined run on 2026-09-14: **877 passed, 1 failed, 3 subtests passed**
(20.89 seconds). Breakdown:

| Suite | Result |
| --- | --- |
| Step 21 dedicated | 62 passed |
| Step 20 bridge | 103 passed |
| Step 19 beliefs | 117 passed |
| Step 18 multiple futures | 144 passed |
| Step 17 cross-modal consequences | 93 passed |
| Experience/prediction memory and learning (four existing files) | 119 passed |
| Physical world / constraints / latent / hybrid / common evidence | 228 passed |
| Additional legacy prediction evaluation | 11 passed, 1 failed, 3 subtests passed |

Reproduce with `.venv/Scripts/python.exe -m pytest -q` followed by
`evaluation/tests/test_astra_efa_experience.py`,
`evaluation/tests/test_belief_prediction_bridge.py`,
`evaluation/tests/test_bayesian_belief_state.py`,
`evaluation/tests/test_multiple_futures.py`,
`evaluation/tests/test_cross_modal_consequences.py`,
`evaluation/tests/test_experience.py`,
`evaluation/tests/test_experience_learning.py`, `test_prediction_experience.py`,
`test_experience_learning.py`, `test_prediction_evaluation.py`,
`evaluation/tests/test_physical_world_model.py`,
`evaluation/tests/test_physics_constraints.py`,
`evaluation/tests/test_latent_physical_state.py`,
`evaluation/tests/test_hybrid_world_state.py`, and
`evaluation/tests/test_common_evidence_state.py`.

The additional legacy live-overlay regression independently reproduces a pre-existing
failure: `test_prediction_evaluation.py::EvaluationTests::test_live_overlay_renders_latest_result`
passes BOM-prefixed `live_app.py` text to `ast.parse`, which raises SyntaxError.
Neither that test nor the protected application file is changed or bypassed.

## 24. Canonical epistemic statements

> Raw history is not the same as experience history.
> An ExperienceEpisode is not ground truth.
> A transformation records change in AISTHESIS internal longitudinal state; it does not assert that the physical world changed in the same way.
> Supported does not mean certain.
> Contradicted does not mean impossible.
> Unobservable and insufficient evidence are not contradictions.
> Missing measurements are not zero.
> Prediction error is preserved as structured evaluation information, not collapsed into a universal loss.
> A trajectory can describe accumulated state change without demonstrating learning or improvement.
> Step 21 does not modify Bayesian beliefs.
> Step 21 does not modify ExperienceLearning.
> Step 21 does not infer hidden actors.
> Step 21 does not promote expected cross-modal consequences into observations.
> Step 21 is an engineering integration of ASTRA–EFA concepts, not validation of the ASTRA–EFA scientific theory.

Experience != truth. Evaluation status != reality. Prediction error != causal
explanation. Posterior != experience or truth. Internal state change != physical
event. Missing evidence != negative evidence. Memory != learning. History !=
adaptation. Adaptation trajectory != biological adaptation. Distinctiveness !=
importance, salience or reward.
