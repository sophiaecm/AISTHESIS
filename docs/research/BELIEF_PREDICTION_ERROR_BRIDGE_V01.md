# Belief → Prediction → Outcome/Error Bridge v0.1 — Step 20

## 1. Purpose

Issue an explicitly selected Step 19 hypothesis as an existing `PredictionRecord`
and compare it with a later observation through the frozen prediction stack.
Preserve the full Step 18/19 source and explain which fields can be evaluated.
This is integration plumbing, not a new evaluator or prediction-error engine.

## 2. Existing frozen machinery reused

- `prediction_records.PredictionRecord`, `OutcomeRecord`, `PredictionEvaluation`.
- `prediction_observation.outcome_from_scene(...)` and `evaluate_prediction(...)`.
- `PredictionEvaluation.as_prediction_error()` for the legacy error container.
- `prediction_observation.experience_episode(...)` and existing `ExperienceEpisode`.
- Existing `EvaluationStatus` values and typed exact-ID association rules.

`prediction_from_hypothesis(...)` was inspected but is not called: it projects
the older hypothesis contract, whereas this bridge projects the actual Step 18
source future retained by Step 19. Both paths produce the same existing
`PredictionRecord` contract. No frozen source file or comparison policy changed.

## 3. Architecture

```text
BayesianBeliefState + explicit hypothesis_id + explicit target/duration
    + optional original LatentPhysicalState for tracker identity
    -> issue_prediction
    -> BeliefPredictionIssue containing PredictionRecord or a non-issued result

issued prediction + later SceneState + explicit session/frame context
    -> outcome_from_scene
    -> evaluate_prediction
    -> experience_episode
    -> BeliefPredictionOutcome
       + detached legacy PredictionError on request
```

There is no feedback arrow to Bayesian belief, confidence calibration or memory.

## 4. Step 19 → issued prediction mapping

```python
from fifth_layer.world_model.belief_prediction_bridge import (
    issue_prediction, evaluate_issue,
)

issue = issue_prediction(
    belief_state,
    hypothesis_id=chosen_hypothesis_id,
    target_timestamp=target_time,
    physical_state=original_latent_state,  # optional, for attributed track binding
)

if issue.issue_status == "issued":
    result = evaluate_issue(
        issue,
        later_scene,
        session_id=belief_state.session_id,
        coordinate_frame_id=belief_state.coordinate_frame_id,
        time_tolerance_seconds=0.0,
    )
    error = result.prediction_error
    episode = result.experience_episode
```

`issue_prediction` validates `BayesianBeliefState` v0.1, finds exactly the named
`HypothesisBelief`, and retains it as `source_belief`. That record includes the
complete `source_future`, branch assumptions, provenance and source references.
No hypothesis is regenerated and no new semantic branch is added.

`BeliefPredictionIssue` is immutable. It contains `issue_id`, `belief_state_id`,
`source_belief`, `hypothesis_id`, `source_future_id`,
`source_posterior_probability`, optional `prediction`, `issue_status`,
`evaluable_fields`, `unevaluable_fields`, `target_timestamp`, `provenance` and
`schema_version=belief-prediction-bridge-v0.1`.

Statuses are `issued`, `unavailable`, `insufficient_prediction_content`.
Missing known source/target time is unavailable. With valid time, no supported
single-object change produces insufficient prediction content. Neither case
fabricates a `PredictionRecord`.

## 5. Explicit branch selection policy

`hypothesis_id` is a required keyword. Blank or unknown IDs are rejected. There
is no highest-posterior selection, threshold, ranking, pruning or MAP decision.
An explicitly chosen low-posterior branch can be issued. Unavailable probability
does not prevent issuing otherwise supported branch content. Source branch
status and probabilities remain recorded; issuance does not strengthen them.

`PredictionRecord.confidence` and scalar `uncertainty` remain `None`.
`evidence_for` and `evidence_against` stay empty: assumptions and likelihood-model
values are not promoted to observational support.

## 6. Qualitative horizon limitation

Step 18's `next_transition` and `short_horizon`, with `horizon_value=None`, supply
no duration. The caller must provide `target_timestamp` or positive
`horizon_seconds`. Supplying both is allowed only when they agree under the
existing absolute `1e-9` timing tolerance.

The issuance source timestamp is the current belief cutoff, not wall-clock time
and not the older latent snapshot timestamp. Target time must strictly follow
that cutoff; duration is `target - source`. The original future timestamp must
also be known. Invalid/nonfinite times are rejected; unknown source time or
absent numeric horizon yields an unavailable, non-issued result.

No qualitative horizon is mapped to arbitrary seconds. If belief was updated
after the source future was created, the original branch remains visible as
older source content; the bridge does not recompute motion or refresh its claims.

## 7. Evaluable versus unevaluable Step 18 fields

| Step 18 change | Bridge handling |
| --- | --- |
| `motion_state=moving`, `epistemic_status=branch_assumption` | Map exactly to `predicted_state={"motion_state": "moving"}` for a single physical object. |
| `motion_state=unresolved` | Unevaluable; never converted to stationary or another motion state. |
| `contact_status` | Unevaluable; no synthetic contact/impact event comparator. |
| `continuity_status` | Unevaluable; no visibility, occlusion or actor interpretation. |
| `consequence_reference` | Unevaluable; never converted to an observed sensory claim. |

Issued motion records use the existing `hypothesis_type=continued_motion`, but
do not add `event` to predicted state. No center, trajectory, speed, exact sound,
temperature, force, visibility, sensory modality or sensory claim is generated.
Mixed content retains all unsupported changes in the source and field inventory
while evaluating only the supported motion projection. Support therefore applies
only to mapped content, not automatically to every source branch assumption.

## 8. New evidence and association semantics

Evaluation accepts an actual later `SceneState` v0.1 or `None` for missing data.
It never accepts a branch or belief as an outcome. The existing
`outcome_from_scene` reads fresh observed object/motion/explicit observation
records. Predicted tracks, stale records, inferred records and expected sensory
consequences are not outcome sources. The bridge does not copy prediction content
into observations or infer absent event keys as false.

Step 18 physical object IDs are not necessarily tracker or observation object IDs.
The optional original latent state must match the future's physical-state ID,
scene, original timestamp, session and frame. For one branch endpoint, an
`observed` or `estimated` attributed `track_id` with source references may bind
the prediction. Duplicate source tracker IDs remain unbound. IDs retain their
types, including integer zero; strings are not coerced to integers.

Without a usable source tracker identity, issuance can still record the selected
motion prediction, but later association is `unbound`. Physical IDs are never
parsed, reused as frame-local observed IDs or resolved by class/nearest geometry.
No object-ID persistence override is enabled by this bridge.

Callers explicitly supply evaluation `session_id` and `coordinate_frame_id`.
They must match the issue, and any declared later-scene or admitted-observation
metadata must agree. This matters because the legacy `SceneState` has no dedicated
session/frame fields. Context assertions and tracker continuity remain caller
responsibilities; matching a typed ID does not authenticate physical identity.

## 9. Existing evaluation semantics reused

The bridge calls the frozen evaluator with unchanged
`time_tolerance_seconds` and `position_tolerance_pixels` arguments, both defaulting
to zero. A later observation outside target-time tolerance yields the existing
insufficient-evidence result; the bridge does not move the target to fit evidence.
The generic moving comparator accepts directional moving observations, rejects
stationary observations and leaves unknown/conflicting motion insufficient.

Existing outcomes remain distinct:

- Matched observed motion can be `supported` or `contradicted`.
- A bound track missing from observations is `unobservable`, not occluded.
- Ambiguous/unbound association or missing/conflicting motion is
  `insufficient_evidence`.
- Missing event keys remain unknown. No contact, sensory or event comparison is
  added by this milestone.

The evaluator still owns `partially_supported` and all other existing statuses.
The current bridge's single motion comparator does not add a new partial-score
policy. Target-state compatibility is not probability calibration.

## 10. PredictionError compatibility

`BeliefPredictionOutcome.prediction_error` calls
`evaluation.as_prediction_error()`. The legacy `PredictionError` container is
mutable, so each access returns a fresh detached instance. Mutating it cannot
alter the immutable issue, evaluation or episode. No parallel error class,
error formula or causal explanation is implemented.

## 11. ExperienceEpisode compatibility

The existing `experience_episode(prediction, outcome, evaluation)` constructs
the returned episode. Its existing summaries preserve exact prediction,
observation and evaluation records. `learning=False` and
`inference_feedback=False` remain intact. The bridge neither stores the episode
in `ExperienceMemory` nor invokes experience retrieval/learning.

`BeliefPredictionOutcome` is an immutable link wrapper around `issue`, `outcome`,
`evaluation` and `experience_episode`. It validates matching record IDs, times
and episode summaries. Convenience properties expose `issue_id`, `prediction`,
existing evaluation `status` and the detached legacy error.

## 12. Provenance

Traceability is preserved through evaluation prediction ID → existing prediction
provenance → issue ID and belief ID → selected hypothesis → complete source future.

Prediction metadata explicitly retains source physical state, constraint,
consequence and field references; source assumptions and predicted changes;
branch family/status; prior/posterior/likelihood as source values; evidence-lineage
IDs; session/frame; source schema versions; association and issuance policies.
The issue additionally retains complete source belief/future, belief provenance
and full evidence lineage. Posterior remains context, never observed truth.

Legacy prediction records validate scalar uncertainty fields. Full structured
Step 18/19 uncertainty/provenance therefore stays in the bounded bridge wrapper;
the legacy record receives a compatible reference projection. This uses existing
summary/reference contracts without weakening or changing them.

Outcome provenance is supplied by the existing observation adapter; evaluation
provenance records its comparison tolerances and interpretation. Both remain
unchanged. The bridge declares `truth_decision=not_performed`.

## 13. Determinism

Existing `stable_id` produces issue and prediction IDs from source identity,
selected content, explicit target, mapping and provenance. Existing functions
produce outcome/evaluation/episode IDs. Field inventories and JSON keys are
sorted; JSON serialization is deterministic and returns detached data.

Metadata uses existing `bounded_plain`/`freeze` rules, including node/depth/string
and serialized-size limits. No random UUID, wall clock or unstable representation
is used. The adapter never sorts hypotheses by posterior.

## 14. Time and future leakage rules

All source belief/future metadata is checked at the issuance cutoff before
constructing a prediction. Optional latent identity evidence must belong to the
original snapshot. Future source timestamps are rejected; only the explicit
prediction target/duration describes future time.

A supplied observation scene must have a known timestamp strictly later than
the prediction source, even if a generous evaluator tolerance is requested.
Unknown, equal or earlier observation times are rejected. `None` represents an
absent scene and can yield an existing missing-evidence evaluation.

Admitted observation provenance, including event timestamps, cannot extend into
the future relative to the later scene. Unused predicted-track metadata is not
reinterpreted as evidence. The issue API has no later-scene argument, and
evaluation does not reconstruct or revise an issued prediction.

## 15. Missing/unknown handling

No numeric horizon means no issued prediction. Unsupported branch content means
insufficient prediction content. Missing tracker binding remains unbound; absent
later tracks remain missing, not occluded. Missing source values, events or
measurements never become zero, false, negative evidence or contradiction.

Every non-issued result preserves the selected source and field inventory.
Non-issued results cannot be passed to `evaluate_issue`. Evaluation returns the
existing statuses without collapsing unobservable/insufficient evidence into
contradiction.

## 16. No automatic Bayesian feedback

Supported does not become likelihood 1; contradicted does not become likelihood
0. The bridge never calls Bayesian initialization/update, alters priors/posteriors,
calibrates models, adjusts experience weights or chooses another branch.
Evaluation of one projection does not validate or invalidate the full label
distribution, and contradiction does not eliminate every alternative branch.

## 17. Explicit non-goals

No new prediction/evaluation/error engine, causal explanation, hidden-actor
inference, automatic branch selection, hypothesis generation, long-horizon
planning, probability calibration, learning, live UI integration or observation
fabrication. No model, package dependency, external repository, camera, GPU,
internet or external dataset is required.

## 18. Testing

Dedicated synthetic tests use actual Step 17–19 sources, latent attributes and
later observed `SceneState` records. They compare bridge results directly with
the existing outcome/evaluator/episode functions. Coverage includes selection,
timing, safe mapping, source references, typed identity, missing/ambiguous/stale
observations, tolerance, frozen records, detached legacy errors, no feedback and
process-level determinism without model imports.

```powershell
.venv/Scripts/python.exe -m pytest evaluation/tests/test_belief_prediction_bridge.py -q
.venv/Scripts/python.exe -m pytest evaluation/tests/test_bayesian_belief_state.py evaluation/tests/test_multiple_futures.py evaluation/tests/test_cross_modal_consequences.py -q
.venv/Scripts/python.exe -m pytest test_prediction_experience.py test_experience_learning.py evaluation/tests/test_experience.py evaluation/tests/test_experience_learning.py -q
.venv/Scripts/python.exe -m pytest evaluation/tests/test_physical_world_model.py evaluation/tests/test_physics_constraints.py evaluation/tests/test_latent_physical_state.py evaluation/tests/test_hybrid_world_state.py evaluation/tests/test_common_evidence_state.py -q
```

Verification: Step 20 103 tests; Step 19 117; Step 18 144; Step 17 93;
existing prediction/experience suites 119; world-model regressions 228.
Existing tests were not weakened. No generated result fixture was edited.

## 19. Validation status

- 🧩 IMPLEMENTED
- 🧪 TESTED
- 📊 NOT YET TASK-LEVEL EVALUATED
- 🏆 NOT YET EXPERIMENTALLY SUPPORTED

## 20. Known limitations

Only generic motion currently maps into a comparator. Contact, continuity,
unresolved motion and candidate sensory consequences remain unevaluable. A source
latent state is needed for actual tracker binding; the branch alone does not
carry sufficient tracker identity. No metric projection or sensor adapter is
added. Retained branch assumptions may predate the current belief cutoff; their
age is preserved, not silently refreshed.

Caller session/frame assertions do not authenticate measurements or track
continuity. Evaluation checks only mapped target-state content, not complete
branch truth or all-world probability calibration. Full provenance is bounded;
oversized records fail validation rather than being silently truncated.

## 21. Relationship to Step 21 ASTRA–EFA Experience Integration

The returned existing `ExperienceEpisode` gives Step 21 a clean evaluated record
to consume later. ASTRA–EFA integration, memory admission, retrieval and learning
are not implemented here. Existing experience-learning behavior remains unchanged.

## 22. Relationship to Experiment 001

Step 20 supplies plumbing for later measurement of whether issued candidate
content is supported or contradicted by observations. It does not demonstrate
hidden-actor anticipation, improved Time-to-Anticipation, prediction accuracy,
calibrated probabilities, cross-modal benefit or scientific support for AISTHESIS.
Experiment 001 requires later ground-truth evaluation; no success is claimed.

## 23. Canonical epistemic statements

> A prediction is not an observation.

> A posterior probability is not truth.

> The highest-posterior branch is not automatically selected.

> An issued branch remains a hypothesis until later evidence is evaluated.

> Missing observation is not contradiction.

> Unobservable and insufficient evidence are not negative evidence.

> A prediction error measures compatibility under an explicit comparison policy; it is not a causal explanation.

> Evaluation of one branch does not validate or invalidate the complete belief distribution.

> Step 20 does not automatically feed evaluation results back into Bayesian belief.

> Step 20 does not infer hidden actors.

> Step 20 does not promote expected cross-modal consequences into observed sensory evidence.

> Step 20 is an engineering integration layer, not scientific validation.

Branch != real event. Selected branch != confirmed branch. Target-state match !=
calibrated probability. Prediction error != scientific falsification of AISTHESIS.
Expected sound != heard sound. Track loss != occlusion. Tracking discontinuity !=
hidden actor. Low posterior != impossible. High posterior != certainty.
