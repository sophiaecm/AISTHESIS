# Bayesian Belief State v0.1 — Step 19

## 1. Purpose

Update an explicit distribution over Step 18 hypothesis labels using numerical
priors and numerical likelihoods supplied by the caller. Keep every Bayesian
term, physical branch, assumption and source reference inspectable. This layer
performs engineering belief updates; it does not supply calibrated probabilities,
sensor models, physical truth or hidden-actor detection.

## 2. Architecture and probability scope

```text
MultipleFutureBundle v0.2
    + explicit prior mapping and source provenance
    OR explicitly requested uniform initialization
    -> BayesianBeliefState (otherwise unavailable)
    + LikelihoodEvidence + caller-declared update time cutoff
    -> deterministic Bayesian update
    -> new BayesianBeliefState with evidence lineage
    -> later prediction/evaluation consumer (not implemented here)
```

Step 18 branches are local, potentially overlapping, non-exhaustive hypotheses.
They are not a partition of real-world events. Step 19 therefore explicitly
models a **categorical variable over branch labels**. Its fixed
`probability_scope=categorical_branch_labels_not_event_frequencies` identifies
that modeling choice. Normalizing these labels does not assert that their
physical contents are exhaustive or mutually exclusive. Probabilities must not
be summed as physical event frequencies when branch contents overlap.

No branch relations are used to manufacture priors or likelihoods. Existing
source contracts and relation semantics remain intact. Only the three Step 19
files are added; frozen producers and Common Evidence State are unchanged.

## 3. Input contracts

`BayesianBeliefStateBuilder.initialize()` accepts the actual
`MultipleFutureBundle` v0.2. All branches are eligible, including `indeterminate`,
`unsupported` and `unavailable`. Their status does not assign a probability.

For each actual `FutureStateCandidate`, the complete frozen record is preserved:

- `future_id`, `scene_id`, `timestamp`, `session_id`, `coordinate_frame_id`.
- `horizon_kind`, `horizon_value`, `status`, `branch_family`, `object_ids`.
- `source_physical_state_id`, `source_constraint_ids`, `source_consequence_ids`,
  `source_field_references`.
- `assumptions`, `predicted_changes`, `uncertainty`, `provenance`, `rule_id`,
  `confidence`, `rule_version`, `schema_version`.

The source bundle's context, schema, uncertainty, provenance and branch relations
are also retained. Step 18 has no bundle ID; Step 19 derives a deterministic
content fingerprint rather than inventing an upstream identifier.

The updater accepts a `BayesianBeliefState`, optional `LikelihoodEvidence`, and
optional explicit `as_of_timestamp`. No CommonEvidenceState, VLM prose, raw
sensor payload or learned representation is interpreted to obtain numbers.

## 4. Output contracts

All records are frozen dataclasses with bounded, detached structured metadata,
tuple collections and deterministic `to_dict()`/`to_json()` methods. Schema and
rule version are `bayesian-belief-state-v0.1`.

`HypothesisBelief` contains hypothesis/context identity, the full `source_future`,
`source_future_id`, assumption ID references, `prior_probability`,
`posterior_probability`, `likelihood`, `update_status`, `source_evidence_ids`,
`provenance` and `confidence=None`. `hypothesis_id` equals the original `future_id`
so priors and likelihoods use one unambiguous key space.

`BayesianBeliefState` contains:

- `belief_state_id`, scene/time/session/frame, `source_multiple_futures_id`,
  fixed `source_multiple_futures_schema`, `source_future_ids`, `beliefs`.
- `initialization_mode`, `update_status`, `normalization_constant`,
  `evidence_event_id`, `previous_belief_state_id`.
- Applied `evidence_lineage`, successful `update_index`, `uncertainty`,
  `provenance`, fixed `probability_scope`, schema and rule versions.

Update status is bounded to `initialized`, `updated`, `unchanged`, `unavailable`,
`indeterminate`, `invalid_evidence`. Duplicate/unknown hypothesis IDs, partial
distributions, context mismatch, nonfinite numbers and non-None confidence are
rejected. A constructed updated state must match its explicit prior, latest
likelihood evidence, normalization and posterior terms.

`initialized` stores the initial distribution in both probability fields as the
current usable belief; `likelihood=None` makes clear that no evidence update took
place. `unchanged` similarly carries the unchanged distribution. Failed updates
retain the last usable values in `prior_probability` and set all
`posterior_probability` values to `None`.

## 5. Prior initialization modes

### Explicit prior

```python
from fifth_layer.world_model.bayesian_belief_state import BayesianBeliefStateBuilder

builder = BayesianBeliefStateBuilder()
# Two-branch illustrative fixture; probabilities are explicitly supplied.
ids = sorted(branch.future_id for branch in futures.branches)
belief = builder.initialize(
    futures,
    priors={ids[0]: 0.6, ids[1]: 0.4},
    prior_provenance={"source_id": "synthetic-example-prior"},
)
```

Supplying priors selects `explicit_prior` unless a mode is explicitly provided.
The mapping must cover exactly every source future ID, with no missing or extra
IDs. Nonempty structured `prior_provenance` is required. Values must be finite
Python int/float numbers in [0, 1]; booleans, strings, `None`, NaN and infinity are
rejected. The sum must equal one within absolute tolerance `1e-12` and zero
relative tolerance. Accepted values are retained, not silently renormalized or
repaired. Empty explicit distributions are rejected.

### Uniform uninformative

```python
belief = builder.initialize(futures, initialization_mode="uniform_uninformative")
```

This opt-in mode assigns `1/n` across all represented branch labels. It means
“no hypothesis is preferred by the initialization mechanism,” not equal
real-world frequency. The mode and meaning are recorded in provenance. Empty
source bundles remain unavailable; no division by zero occurs.

### Unavailable

```python
belief = builder.initialize(futures)
```

No prior and no explicit uniform request yields unavailable probabilities, all
`None`. Likelihoods cannot initialize missing priors. Mixing a supplied prior
mapping/provenance with uniform or unavailable mode is rejected.

## 6. Explicit likelihood evidence and time

`LikelihoodEvidence` contains `evidence_id`, scene/time/session/frame context,
`likelihood_by_hypothesis`, nonempty `provenance`, `evidence_type`, bounded
`status` and schema version. Values are explicit finite numbers in [0, 1].
They need not sum to one across hypotheses: these are likelihoods, not a prior.

```python
from fifth_layer.world_model.bayesian_belief_state import LikelihoodEvidence

evidence = LikelihoodEvidence(
    evidence_id="example-event-1",
    scene_id=belief.scene_id,
    timestamp=3.0,
    session_id=belief.session_id,
    coordinate_frame_id=belief.coordinate_frame_id,
    likelihood_by_hypothesis={ids[0]: 0.8, ids[1]: 0.2},
    provenance={"source_id": "synthetic-example-likelihood-model"},
)
# The caller declares that evidence up to this time is available.
posterior = builder.update(belief, evidence, as_of_timestamp=3.0)
```

This example assumes the prior cutoff is at or before 3.0. Scene/session/frame
must match exactly, including `None` frames. Scene identity remains the hypothesis
origin scene; later observations must explicitly refer to this hypothesis set,
not silently rebind it to a different scene.

By default, the cutoff stays at the previous belief timestamp. Later evidence
requires an explicit `as_of_timestamp`. Cutoffs cannot move backward; evidence
cannot predate the previous cutoff or exceed the current cutoff. Nested source
timestamps cannot exceed their containing evidence time. Existing recursive
session/frame/time alignment checks are reused; no wall-clock time is consulted.

Updates require known original source, previous-belief and evidence timestamps.
Unknown time returns `unavailable` with `temporal_alignment_unknown`; advancing a
cutoff does not repair an undated source. Equal evidence timestamps are allowed
for distinct explicit events. A cutoff may advance without an update, so callers
must not advance it beyond evidence they still intend to apply.

Evidence status is `supplied`, `unavailable` or `invalid`. The latter two require
empty likelihood mappings. Unavailable evidence leaves usable belief unchanged;
caller-marked invalid evidence produces `invalid_evidence` without a posterior.
Malformed numeric input is rejected at construction, not accepted as an invalid
number payload. An externally declared model provenance is retained as supplied
metadata, never verified or relabeled as calibrated by Step 19.

## 7. Bayesian update formula

For explicit prior `p_i` and supplied likelihood `l_i`:

```text
w_i = p_i * l_i
Z   = sum(w_i)
posterior_i = w_i / Z, only when representable and Z > 0
```

For the illustrative prior `(0.6, 0.4)` and likelihoods `(0.8, 0.2)`,
`Z=0.56` and the posterior is approximately `(0.8571428571, 0.1428571429)`.
These are synthetic arithmetic results, not measured frequencies.

The implementation uses only Python's standard library. A fixed 80-digit
`decimal` context computes products/normalization from the exact supplied float
values, avoiding premature float-product underflow and isolating caller decimal
settings. Public values remain floats; unrepresentable positive results return
an explicit failure instead of silently becoming zero. This is numerical
engineering, not calibration or inference of missing values.

## 8. Missing-likelihood policy

Default and only policy: every active hypothesis requires an explicit likelihood,
including branches with explicit zero prior. Missing entries yield `unavailable`
with reason `incomplete_likelihoods_no_update`. Missing likelihood fields remain
`None`; supplied entries are retained for audit. Unknown IDs are rejected.
No partial Bayesian update is performed and no absent entry becomes zero.

## 9. Zero-normalization behavior

`Z=0` yields `indeterminate`, `normalization_constant=0.0`, no posterior values,
and retained prior values. The event is not added to applied lineage. No division,
uniform fallback or branch elimination occurs.

If positive normalization cannot be represented as a float, normalization stays
`None` and reason is `normalization_not_representable`. If a positive posterior
would underflow to zero, reason is `posterior_not_representable`; the entire
posterior is withheld. These cases are distinct from mathematically zero input
weights. Explicit zero priors/likelihoods are valid and retain their supplied
meaning; zero probability is not a proof of physical impossibility.

## 10. Sequential update semantics

Sequential updates are supported. A successful posterior becomes the next
update's prior; after a failed attempt, the last usable prior remains available.
Every output references `previous_belief_state_id`. Original prior provenance,
initial mapping and source branch records remain available. Full successful
likelihood records are retained in order in `evidence_lineage`; `update_index`
counts successful updates only.

An already-applied `evidence_id` is rejected, even if the caller changes its
numbers. There is no repeat override. Failed attempts do not consume an event ID,
so a missing-likelihood attempt can be corrected explicitly at a valid cutoff.
The latest attempted evidence and failure reason remain in output provenance.

Each supplied likelihood is interpreted as conditional on the already-applied
evidence. Step 19 does not establish conditional independence. Callers must
avoid correlated evidence double-counting under different IDs; the layer cannot
authenticate distinct physical events from IDs alone. No automatic learning,
experience retrieval, memory mutation or correctness-based calibration occurs.

## 11. Provenance

Each probability record retains its complete Step 18 source future and assumption
IDs. Each initial prior retains caller provenance or the explicit uniform policy.
Each attempted update records previous state identity, the prior field used,
likelihood source/provenance and reason. Successful event IDs appear on every
hypothesis and in ordered state lineage.

The state retains Step 18 bundle schema and content fingerprint, source relations,
source provenance and original source timestamp. No source constraint or
consequence is transformed into a new observation. Source raw records remain
separate from generated belief values. Provenance states
`truth_decision=not_performed`.

Existing `bounded_plain` limits apply: 50,000 traversal nodes, depth 24,
16,384-character strings and a 1 MiB serialized structure. Opaque payloads and
raw model/tensor fields are rejected; provenance is not silently truncated.

## 12. Determinism

Source content, priors, evidence values, provenance, context and previous-state
lineage determine IDs through the existing `stable_id` helper. Hypotheses are
sorted by original future ID, never by posterior. Maps serialize with sorted
keys and compact JSON separators. Evidence lineage preserves application order.
Equivalent mapping/branch permutations produce identical output. Reapplying the
same input to the same immutable prior produces the same result; this differs
from attempting to apply an event twice sequentially.

No UUID, wall clock, unstable representation, random number, softmax, winner,
ranking or MAP decision is used.

## 13. Unknown/missing handling

No evidence leaves initialized belief unchanged. Uninitialized belief remains
unavailable. Missing likelihoods, unknown time and missing prior each have an
explicit reason and do not fabricate a posterior. Failed updates preserve the
last usable distribution in prior fields. Source branch status never supplies a
number: `indeterminate` is not 0.5, and `unsupported` is not zero.

Missing cross-modal consequences or sensors do not become negative evidence.
The numerical updater does not inspect labels, descriptions, class names,
learned metrics, branch confidence, source status or consequence modality to
generate a probability.

## 14. Explicit non-goals

No empirical calibration, sensor-model learning, hidden-actor detector, physical
truth decision, branch winner, hypothesis creation, long-horizon planning,
experience learning, Common Evidence adapter, supports/contradicts assignment,
outcome scoring or prediction-error loop. No external Bayesian framework,
package dependency, model or repository is installed. Camera/photo/video,
Scientific Mini-Lab and existing production components remain unchanged.

## 15. Testing

Tests use actual Step 18 branches and Step 17 candidates with synthetic numerical
models. Coverage includes contract validation, direct arithmetic consistency,
two/three-hypothesis Bayes updates, normalization, zero and tiny numbers, missing
inputs, explicit modes, sequential lineage, duplicate-event rejection, temporal
leakage checks, provenance, immutability and process-level determinism. A fresh
process check confirms no numpy, torch, transformers, PyMC, Pyro or cv2 imports.

```powershell
.venv/Scripts/python.exe -m pytest evaluation/tests/test_bayesian_belief_state.py -q
.venv/Scripts/python.exe -m pytest evaluation/tests/test_multiple_futures.py -q
.venv/Scripts/python.exe -m pytest evaluation/tests/test_cross_modal_consequences.py -q
.venv/Scripts/python.exe -m pytest evaluation/tests/test_physical_world_model.py evaluation/tests/test_physics_constraints.py evaluation/tests/test_latent_physical_state.py evaluation/tests/test_hybrid_world_state.py evaluation/tests/test_common_evidence_state.py -q
```

Implementation verification: 117 Step 19 tests, 144 Step 18 tests, 93 Step 17
tests and 228 relevant world-model regression tests passed. Existing tests were
not weakened. These are engineering checks, not a task-level benchmark.

## 16. Validation status

- 🧩 IMPLEMENTED
- 🧪 TESTED
- 📊 NOT YET TASK-LEVEL EVALUATED
- 🏆 NOT YET EXPERIMENTALLY SUPPORTED

## 17. Known limitations

Probabilities are conditional on caller-supplied numerical models and the chosen
categorical label space. Branch overlap, incomplete coverage and duplicate
representations can make label probabilities unsuitable as real-world event
probabilities. There is no likelihood-model verification or calibration. Prior
zero values cannot acquire positive mass through ordinary multiplicative updates.

Sequential event dependence and genuine event identity remain caller concerns.
History grows within bounded metadata limits and eventually requires a separately
designed retention policy; no silent truncation is performed. Scene/frame changes
and undated source sequences require new compatible source initialization. The
layer does not transport beliefs across changed hypothesis sets or frames.
Extreme unrepresentable float outputs are explicitly indeterminate.

## 18. Relationship to Step 18

Step 18 remains a probability-free physical hypothesis layer. Step 19 retains
every supplied branch and its ambiguity, including source consequence statuses.
It adds a caller-defined distribution over labels and never retroactively changes
Step 18 confidence, assumptions, branch relations or future descriptions. The
existing Step 18/17 documents and older source APIs/regression suites were
inspected; no compatibility edits to frozen files were needed.

## 19. Relationship to Step 20 and experience learning

Step 20 can later consume belief states in a prediction → new evidence → outcome
comparison → prediction-error loop. None of those operations is implemented
here. Existing experience-learning components remain separate: no priors or
likelihoods are learned from history, no experience memory is modified or
retrieved to bias initialization, and no past correctness calibrates probabilities.

## 20. Relationship to Experiment 001

A later evaluation may compare **Vision + Physics** with **Vision + Physics +
Cross-Modal Consequences**, using belief updates over future hypotheses. Step 19
does not demonstrate hidden-actor anticipation, better Time-to-Anticipation,
predictive accuracy, calibrated probabilities or scientific support for AISTHESIS.
Those claims require ground-truth task evaluation. Experiment 001 success is not
claimed.

## 21. Canonical epistemic statements

> Posterior probability is not truth.

> The highest-probability hypothesis is not automatically the real future.

> Missing evidence is not evidence against a hypothesis.

> Missing likelihood is not zero likelihood.

> Uniform prior initialization is not a claim about real-world frequency.

> Likelihood values in Bayesian Belief State v0.1 are explicitly supplied; they are not learned or calibrated by this milestone.

> Bayesian normalization does not establish empirical calibration.

> Step 19 does not infer hidden actors.

> Step 19 does not convert cross-modal consequences into observations.

> Step 19 is an engineering belief-update layer, not a scientifically validated predictor.

Posterior != truth. Highest posterior != reality. Probability != observation.
Prior != fact. Likelihood != certainty. Low probability != impossible. Missing
evidence != evidence against. No update != negative evidence. Normalization !=
calibration. Bayesian consistency != empirical validity. Equal prior != claim
of equal real-world frequency. Supplied prior != learned prior. Supplied
likelihood != verified sensor model. Future branch != real event. Cross-modal
consequence != observed sensory event. Learned signal != physical fact. Tracking
discontinuity != hidden actor.
