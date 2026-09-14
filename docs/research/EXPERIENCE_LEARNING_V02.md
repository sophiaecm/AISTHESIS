# Step 22 — Experience Learning v0.2

## 1. Purpose

Provide an explicit, opt-in, bounded historical-context view of current Step 19
branches using completed Step 21 experiences. This is engineering integration;
the term experience learning does not establish learning success.

```python
from fifth_layer.world_model.experience_learning_v02 import experience_learning_v02

disabled = experience_learning_v02(belief_state)  # no trajectory access
view = experience_learning_v02(
    belief_state, trajectory, enabled=True,
    top_k=3, min_similarity=0.5, max_absolute_contribution=0.05,
)
```

Only `experience_learning_v02.py`, its dedicated test file, and this document are
added. No application, Mini-Lab or frozen module is modified.

## 2. Relationship to Experience Learning v0.1

The v0.1 API and behavior are unchanged. v0.2 adds a separate API for Step 19
beliefs and Step 21 trajectories; it does not call v0.1, replace its records, use
its memory retrieval, or insert EvidenceItems. It shares the conservative
absolute/relative bounds and explicit unknown semantics. v0.2 uses the requested
status-direction mean over retrieved structural context. Unlike v0.1's event
metric filter, structural conflicts can lower similarity without excluding a
record that still meets the retrieval gates. This policy is heuristic and is
exposed in every comparison and contribution audit.

## 3. Relationship to ASTRA–EFA Step 21

Step 21 remains an immutable representation of episodes and internal changes.
v0.2 reconstructs its trajectory from the detached source episodes and verifies
the initial state, encounters, transformations, pre/post states, histories and
generated IDs. Malformed lineage raises ValueError; it is never repaired.
Transformation IDs are context/provenance, not a measure of learning magnitude.

New frozen contracts:

| Contract | Content |
| --- | --- |
| `TrajectoryExperienceQuery` | Current scene/session/time, belief/hypothesis/future IDs, branch family, assumption types, predicted fields, constraint/consequence IDs, structural features and provenance |
| `RetrievedTrajectoryExperience` | Query/experience/episode/prediction IDs, historical status, similarity and feature comparisons, historical timestamps/metrics, transformation/pre/post IDs, sequence and provenance |
| `ExperienceLearningRow` | Hypothesis/future IDs, base posterior, contribution, heuristic score, influence status, retrieved records and derivation provenance |
| `ExperienceLearningViewV02` | View ID, scene/session/time, source belief/trajectory IDs, enabled flag, queries, rows, retrieval audit and provenance |
| `TrajectoryExperienceRetriever` | Deterministic structural retrieval with explicit top-k and threshold |

All records use schema `experience-learning-v0.2`. Queries, retrievals and views
have content-derived stable IDs. Rows are uniquely ordered by hypothesis ID.

## 4. Current-query construction

One query per current HypothesisBelief is built from its source FutureStateCandidate.
The API requires BayesianBeliefState v0.1, rechecks the frozen contracts, and
validates declared current source timestamps/session/frame alignment.

Exact semantic feature inventory:

| Feature | Current extraction | Historical extraction |
| --- | --- | --- |
| `branch_family` | Source future branch family | `source_branch_family` |
| `assumption_types` | Sorted unique assumption type labels | Types from `source_assumptions` |
| `predicted_fields` | Sorted unique recognized change field names | Names from `source_predicted_changes` |
| `predicted_values` | Sorted physical `(field, value, epistemic_status)` tuples | Same projection of stored changes |
| `horizon_kind` | Explicit future horizon kind | `source_horizon_kind` |
| `object_count` | Number of referenced physical objects | Unavailable (`None`) in this bridge contract |
| `has_consequence_references` | Whether explicit consequence ID tuple is nonempty | Boolean from explicit `source_consequence_ids`; missing is `None` |

Full current predicted changes, object scope/IDs, source schemas, prior/posterior,
source evidence IDs and source provenance remain query provenance only.
No prior, posterior, likelihood, tracker ID, object ID, scene ID, prediction ID,
future ID or constraint ID is a semantic matching feature.

## 5. Historical feature extraction

Read only the stored prediction provenance in the Step 21 encounter. Do not use
later outcome state, evaluation metrics, status frequencies or transformation
magnitude as similarity features. The status is used only after retrieval for
direction. Current feature content cannot fill missing historical fields.

Recognized physical values are `motion_state=moving`,
`contact_status=possible_contact|no_contact_assumed`, and
`continuity_status=possible_continuity`, each with `branch_assumption` status.
`unresolved` values are unavailable for value matching. `consequence_reference`
may appear in the field inventory, but its ID value is excluded from semantic
values. Expected cross-modal reference values are never observations.

Assumption types use the Step 18 vocabulary: motion_persists,
contact_within_horizon, no_contact_within_horizon, continuity_resumes and
outcome_unresolved. Missing or malformed type structures are unavailable.
Historical object count is not inferred from tracker binding or opaque IDs.
Constraint types and evaluable-field matching are deferred: no reliable symmetric
contract is available for all supplied current and historical records.

## 6. Eligibility policy

When enabled, an actual AstraExperienceTrajectory v0.1 is required; `None` or an
incompatible type/version raises ValueError. Its entire lineage is revalidated
before retrieval. Each record must be in the query session and strictly past:
prediction source, prediction target, observation and evaluation timestamps must
all be less than query time. Any explicit predicted event timestamp must also be
strictly past. This is deliberately stricter than evaluation time alone, following
v0.1's future-target protection. Step 21 already rejects incomplete, nonterminal,
inconsistent or future-source records.

Current-time/future records are audited as `not_strictly_past`; unknown query
time is `unknown_query_timestamp`. They supply no retrieval or contribution.
Broken trajectory objects are rejected as a whole, not partially consumed.

## 7. Similarity policy

Every fixed feature is marked `matched`, `conflicting`, or `unavailable` with its
current/historical values. `None` is unavailable and excluded from the denominator.
Exact type-consistent equality is required. Similarity is:

`matched_count / (matched_count + conflicting_count)`

With no comparable features, similarity is zero. Retrieval additionally requires:

- A matching branch family.
- Matching physical predicted values, or matching assumption types other than
  the sole generic `outcome_unresolved` type.
- Similarity at least `min_similarity`, default **0.5**, caller range **[0, 1]**.

Family, horizon, object count, field names or absence of consequence references
alone cannot pass the substantive-match gate. Thresholds have no scientific
significance. Default top-k is **3**; the permitted integer range is **1–3**.
Sort by descending similarity, then descending evaluation time, then experience
ID. Time is a tie-break only, with no weighting or automatic decay.

## 8. Status-direction policy

| Historical status | Direction |
| --- | ---: |
| supported | +1 |
| contradicted | -1 |
| partially_supported | 0 |
| unobservable | 0 |
| insufficient_evidence | 0 |

All five are terminal. Other statuses cannot enter a valid Step 21 trajectory.
Neutral records never create a positive or negative term. They remain in the
retrieved mean's denominator, so they can attenuate a mixed nonzero signal.
No match and missing information are not counterevidence.

## 9. Bounded influence

For base posterior `p`:

```text
bound = min(max_absolute_contribution, 0.05 * p)
term_i = direction(status_i) * similarity_i
mean_term = sum(term_i) / number_of_retrieved_records
contribution = clamp(bound * mean_term, -bound, +bound)
experience_informed_score = p + contribution
```

The mean is zero with no retrieved records. `math.fsum` calculates the numerator.
The configurable absolute maximum defaults to **0.05** and accepts **[0, 0.05]**.
The fixed relative maximum is **5% of base posterior**. Bools, nonfinite values
and out-of-range settings are rejected. No linear accumulation with history
length occurs. Top-k, averaging, and clamping bound even repeated similar context.

## 10. Why experience-informed score is not probability

It is a separate additive heuristic and may exceed one when a base of one receives
positive context. Rows are never normalized and need not sum to one. They remain
ordered by hypothesis ID, not score. No winner is selected. The original posterior
is displayed unchanged under `base_posterior_probability`.

## 11. Why no posterior mutation occurs

The API reads immutable source records and creates detached output records. It
does not write beliefs, futures, episodes, memory or trajectories. No writable
source mapping is attached to output. Source objects remain structurally and
serially identical. Unknown posterior gives contribution zero and score `None`;
zero posterior has bound zero and cannot be revived by history.

## 12. No automatic Bayesian feedback

No Bayesian builder update is called, no LikelihoodEvidence is constructed,
no frequency becomes a prior and no probability renormalization occurs. Existing
likelihoods/evidence lineage are neither modified nor interpreted as historical
direction. No future generation or branch selection is called.

## 13. Current-evidence dominance

Contribution is suppressed for unavailable, indeterminate or invalid-evidence
belief states; unknown query timestamps; unsupported, unavailable or indeterminate
source futures; and missing posteriors. The explicit frozen Step 17/18 uncertainty
marker `source_constraint_conflict_or_ambiguity` also suppresses contribution.
This is an exact known marker check, not free-text evidence interpretation.

Step 19 has no general signed current support/oppose inventory. v0.2 does not
invent one, reinterpret small likelihoods as opposition, or infer opposition
from arbitrary constraint IDs. Source statuses and the explicit conflict marker
are the available gates. Retrieval remains auditable when a current-state gate
suppresses scoring. No relevant history yields zero with `no_relevant_history`
unless a stronger current-state suppression reason applies.

## 14. Raw history vs trajectory context

Step 21 raw episode history remains separate from its transformation history.
Each retrieval links episode, prediction, evaluation, outcome, encounter,
transformation, pre/post state and sequence. No history is rewritten; counters,
transformation magnitude and state deltas are not scoring features.

## 15. Same-session restriction

v0.2 is same-session only. A valid different-session trajectory produces explicit
`session_mismatch` audit entries and zero contribution. No cross-session identity
or environmental compatibility is assumed. Cross-session retrieval is deferred.

## 16. Missing/unknown semantics

Missing features are `None`, not zero. Explicit empty consequence references are
known absence, distinct from an absent metadata field. Expected values are not
measurements. Missing posterior is never numerically filled. Empty history has
zero contribution. Enabled mode without a trajectory raises ValueError.

Disabled mode is the default. It builds current-only queries and rows, contributes
zero, performs no history validation/retrieval, and does not even read a supplied
trajectory ID; `source_trajectory_id=None` and `retrieval_audit=()` make this explicit.

## 17. Deduplication

Step 21 lineage rejects repeated experience/episode/prediction IDs. Retrieval also
tracks those IDs and outcome IDs independently, so differently represented copies
of one Step 20 outcome do not become independent terms. First occurrence in valid
trajectory order owns the identity. Repeated outcome IDs are audited as
`duplicate_history`. Correlation beyond these explicit identifiers cannot be
established; this is another reason to retain the small hard bound.

## 18. Determinism

Stable IDs use canonical structured values and sorted JSON, without wall-clock,
randomness or repr-based identities. Feature order, query/row order, retrieval
ties, arithmetic and audits are deterministic. Audit order follows trajectory
order. Adding provenance can change IDs without changing semantic similarity.
No automatic recency decay is applied.

Existing bounded serialization limits and Step 21's growing snapshots apply;
this is intended for small session trajectories. It is not an unbounded store or
an optimized large-history index. Lineage validation can inspect later records
to reject malformed input, but later records never contribute to earlier-query
similarity, top-k or scores. The view's source trajectory ID remains provenance.

## 19. Auditability

Each considered experience has query/experience/episode/prediction/outcome IDs,
historical timestamps, eligibility, retrieval decision and reason. Structurally
compared records include all feature components and similarity even when below
threshold or outside top-k. Invalid top-level lineage fails before scanning.

Retrieved records preserve exact historical metrics, prediction/evaluation
provenance, transformation/pre/post IDs and sequence. Each row records its bound,
absolute/relative settings, directional terms, mean, formula, current statuses
and suppression reason. Provenance explicitly declares:

```text
calibrated=False; probability_update=False; Bayesian_feedback=False
winner_selected=False; history_is_ground_truth=False; current_observation=False
experience_learning_mode="bounded_historical_context"
truth_decision="not_performed"; similarity_is_probability=False
score_is_probability=False; recency_decay=False
```

## 20. Explicit non-goals

No training, reinforcement learning, weight learning, sensor promotion, hidden
actor inference, current evidence bundle insertion, probability calibration,
scientific validation, automatic Bayesian feedback, memory mutation, branch
selection or dynamic reasoning routing. No new dependencies, packages, models,
repositories, datasets, internet access or GPU work.

## 21. Relationship to Step 23 Dynamic Reasoning Connectome v0.2

Step 23 may later route reasoning using separate current physical, Bayesian,
cross-modal, future and bounded historical-context signals. Step 22 does not
implement that routing or wire this view into application inference.

## 22. Relationship to Experiment 001

No improved prediction, TTA, accuracy, hidden actor anticipation, calibration,
cross-modal benefit, adaptation, learning success or Experiment 001 success is
established. Controlled evaluation must compare disabled and enabled behavior
under the same inputs and ground truth.

## 23. Validation status

- 🧩 IMPLEMENTED
- 🧪 TESTED (contract/regression tests; no task-level claim)
- 📊 NOT YET TASK-LEVEL EVALUATED
- 🏆 NOT YET EXPERIMENTALLY SUPPORTED

Final combined run on 2026-09-14: **957 passed, 1 failed, 3 subtests passed**
in 28.22 seconds, using the existing `.venv/Scripts/python.exe`.

| Suite | Result |
| --- | --- |
| Step 22 dedicated | 80 passed |
| Step 21 trajectory | 62 passed |
| Step 20 bridge | 103 passed |
| Step 19 beliefs | 117 passed |
| Step 18 futures | 144 passed |
| Step 17 consequences | 93 passed |
| Existing memory/experience/prediction and learning (four files) | 119 passed |
| Relevant physical/constraint/latent/hybrid/common-evidence regressions | 228 passed |
| Additional legacy prediction evaluation | 11 passed, 1 failed, 3 subtests passed |

The final combined command is:

```powershell
.\.venv\Scripts\python.exe -m pytest evaluation/tests/test_experience_learning_v02.py evaluation/tests/test_astra_efa_experience.py evaluation/tests/test_belief_prediction_bridge.py evaluation/tests/test_bayesian_belief_state.py evaluation/tests/test_multiple_futures.py evaluation/tests/test_cross_modal_consequences.py evaluation/tests/test_experience.py evaluation/tests/test_experience_learning.py test_prediction_experience.py test_experience_learning.py test_prediction_evaluation.py evaluation/tests/test_physical_world_model.py evaluation/tests/test_physics_constraints.py evaluation/tests/test_latent_physical_state.py evaluation/tests/test_hybrid_world_state.py evaluation/tests/test_common_evidence_state.py -q
```

The additional legacy
prediction-evaluation suite reproduces the known Step 21 regression exception:
`test_prediction_evaluation.py::EvaluationTests::test_live_overlay_renders_latest_result`
fails when its helper reads the existing BOM-prefixed `live_app.py` and passes
that string to `ast.parse`. The protected application and old test stay unchanged.

## 24. Canonical epistemic statements

> Past experience is historical context, not a current observation.

> Historical support is not current support.

> Historical contradiction is not current contradiction.

> Similarity is a heuristic retrieval measure, not probability.

> Experience contribution is not a Bayesian posterior update.

> Experience-informed score is not a calibrated probability.

> Missing historical features are unavailable, not zero.

> Unobservable and insufficient-evidence episodes contribute no directional signal.

> Partially supported episodes are neutral in v0.2.

> Repeated similar history cannot exceed the explicit contribution bound.

> Current explicit evidence is not overridden by historical context.

> Step 22 does not mutate the Bayesian belief state.

> Step 22 does not create likelihood evidence.

> Step 22 does not select a winning branch.

> Step 22 does not infer hidden actors.

> Step 22 does not promote historical cross-modal expectations into current sensory observations.

> Step 22 is an engineering historical-context layer, not scientific validation of learning.

Experience != truth. Memory != ground truth. Similarity != causal relevance.
Retrieval score != confidence or likelihood. Historical frequency != prior.
Supported != certain. Contradicted != impossible. No match != counterevidence.
Trajectory change != learning success. Past transformation != current world transformation.
