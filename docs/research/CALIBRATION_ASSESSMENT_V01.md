# Step 27B — Calibration Assessment v0.1

## Status and scientific boundary

🧩 IMPLEMENTED

🧪 TESTED — dedicated, relevant regressions, and full automated suite passed

📊 NOT YET TASK-LEVEL EVALUATED

🏆 NOT YET EXPERIMENTALLY SUPPORTED

**ASSESSABLE != WELL CALIBRATED**

**INSUFFICIENT EVIDENCE != BAD CALIBRATION**

**CALIBRATION RELIABILITY != TRUTH PROBABILITY**

**GOOD CALIBRATION != HIGH ACCURACY**

**HIGH ACCURACY != GOOD CALIBRATION**

Step 27B does not scientifically validate calibration. Unit and regression tests
establish engineering behavior only. This milestone is not committed or tagged.

## Purpose

The assessment asks what empirical evidence supports an explicitly supplied
calibration state, which population that evidence describes, whether its context
supports assessment at issuance, and what remains unknown. It does not estimate
how likely a prediction is to be physically true.

Calibration evidence, calibration claims, prediction correctness, and physical
truth remain distinct. This additive library does not change frozen producers,
Step 27A, application behavior, or any existing test.

## Repository inspection and existing mechanism

Inspection covered `fifth_layer/confidence_calibration.py`,
`fifth_layer/prediction_evaluation.py`, their tests, Step 27A implementation/tests,
Bayesian Belief State and its documentation, Belief Prediction Error Bridge,
Multiple Futures, Experience Learning, and Dynamic Reasoning Connectome.
Searches covered calibration, calibrated confidence, calibration reliability,
sample counts, raw confidence, prediction outcomes, correct/partial/incorrect,
evaluable, posterior, probability, reliability, Brier, ECE, and NLL.

The existing `ConfidenceCalibrationMemory` already provides:

- Outcome weights: correct `1.0`, partially correct `0.5`, incorrect `0.0`.
- Smoothed bucket reliability with the existing alpha/beta constants.
- A five-evaluable-sample activation threshold.
- Source/type/class, source/type, then source fallback.
- A fixed `global_prior` when no eligible empirical bucket is available.
- Bounded influence on raw confidence, bounded memory, replay protection,
  session reset, and TTL behavior.
- `calibrate()` output containing exactly `raw_confidence`,
  `calibrated_confidence`, `calibration_reliability`, `calibration_samples`,
  and `calibration_bucket`.
- `statistics()` output containing one row per retained empirical bucket.

Step 27B changes none of those formulas, constants, policies, or thresholds.
It does not call the memory, even for reads. Callers supply detached public outputs.

`PredictionEvaluationMemory` has five finalized result statuses: `correct`,
`partially_correct`, `incorrect`, `unevaluable`, and `expired`. Pending predictions
are retained separately, not emitted with a finalized `pending` status.
`compare_prediction()` alone can produce a comparison before memory finalization;
a comparison result is therefore not automatically evidence of finalization.

Bayesian distributions remain categorical branch-label models. The belief bridge
preserves their provenance without promoting posterior into prediction confidence.
Experience similarity and routing relevance are heuristic context, not calibration.
Step 27A inventories these heterogeneous meanings without pooling them.

## Architecture and what Step 27B adds

```text
Explicit issuance calibration claim
    + same-session statistics snapshot at issuance, when supplied
    + explicitly finalized historical outcome subset, when supplied
        -> typed, immutable adapters
        -> CalibrationAssessment
            status and named evidence policy
            actual bucket and copied issuance values
            matching empirical statistics, when available
            separate descriptive outcome counts
            limitations, context, lineage, and assumptions
```

Production: `fifth_layer/world_model/calibration_assessment.py`.

The implementation distinguishes reported sample counts from supporting
statistics. It checks declared session/time relationships, detects discrepancies
between issuance and statistics, and reports count sufficiency under a named
engineering policy. It never fits or executes a calibration algorithm.

## Public data contracts

All records are frozen dataclasses. Mappings and sequences are defensively
detached. Nested values are immutable. `to_dict()` returns detached JSON data;
`to_json()` uses canonical key ordering and finite numbers. Fixed wrapper schema
and rule version: `calibration-assessment-v0.1`.

| Record | Content |
| --- | --- |
| `CalibrationContext` | Optional session, timestamp, scene, coordinate frame; attribution explicitly `caller_supplied` |
| `CalibrationIssuance` | Prediction ID, source, type, optional class, optional exact calibration mapping, source reference, provenance, optional forecast creation time, content ID |
| `CalibrationBucketStatistics` | Exact public bucket row: total evaluable, correct/partial/incorrect counts, weighted success, reliability, last-update time |
| `CalibrationStatisticsSnapshot` | Capture context, canonical tuple of typed rows, source reference/provenance, deterministic snapshot ID |
| `FinalizedCalibrationOutcome` | Detached existing evaluation result with nested original prediction and observation, deterministic content reference |
| `CalibrationOutcomeEvidence` | Capture context, sorted finalized outcomes, source reference/provenance, explicit finalization attribution and subset coverage |
| `CalibrationAssessment` | Issuance, optional evidence records, derived status/reasons, copied selected statistics, separate outcome summary, references, assumptions, provenance, versions and policy |

The existing dict-shaped producers do not expose native schema versions or
scene/session/capture IDs. Step 27B does not invent them. `source_contract` names
the public origin; the wrapper schema belongs only to Step 27B. Source timestamps
and source references must be explicitly supplied where unavailable upstream.

The assessment retains issued values under `assessment.issuance.calibration`.
Its copied `empirical_statistics` and derived `outcome_summary` are separate fields;
neither replaces those issued values. Missing calibration metadata is `None`.
Missing statistical or outcome evidence stays `None`, not a zero-valued estimate.

## Explicit adapters

- `adapt_calibration_issuance`: exact `calibrate()` output, or explicit `None`,
  with caller-supplied prediction identity and issuance context.
- `adapt_live_prediction`: exact current `PredictionEvaluationMemory.record_state`
  output; preserves the full original prediction in provenance.
- `adapt_calibration_statistics`: exact `statistics()` rows with caller capture
  context and source reference. Accepts an explicitly empty list.
- `adapt_finalized_outcomes`: existing finalized evaluation results from memory
  history/results, with explicit `finalized=True`, capture context and reference.

`CalibrationAssessmentBuilder.build()` accepts only the typed issuance/evidence
records. There is no permissive producer-dictionary dispatcher. Bayesian objects,
grounding records, uncertainty entries, and their unrelated dict formats are
rejected. New or missing fields in the supported producer contracts fail explicitly.

Example using public outputs already captured by a caller:

```python
from fifth_layer.world_model.calibration_assessment import (
    CalibrationContext, CalibrationAssessmentBuilder,
    adapt_live_prediction, adapt_calibration_statistics,
)

# issued_prediction and statistics_at_issuance are existing public outputs.
# Capture them from the same session/state at the declared issuance time.
context = CalibrationContext(session_id="session-1", timestamp=20.0)
issued = adapt_live_prediction(
    issued_prediction, context=context, source_reference="prediction-log:p1")
statistics = adapt_calibration_statistics(
    statistics_at_issuance, context=context, source_reference="calibration-log:20")
assessment = CalibrationAssessmentBuilder().build(issued, statistics=statistics)
```

The caller controls capture and attribution; this layer cannot independently
verify atomic capture, session identity, upstream authenticity, or finalization.

## Assessment statuses and sufficiency policy

Named policy: `issuance_bucket_evidence_minimum_v0.1`.
Its `minimum_samples` directly imports the existing `MIN_CALIBRATION_SAMPLES=5`.
There is no new numeric threshold.

| Status | Meaning |
| --- | --- |
| `assessable` | Known same-session context, contemporaneous statistics snapshot, matching selected bucket count/reliability, and the existing engineering threshold met |
| `insufficient_evidence` | Source-reported count below the existing activation threshold, including zero; no poor-calibration claim |
| `unavailable` | No issuance calibration claim supplied |
| `indeterminate` | A sufficient-count claim lacks corroborating statistics, context is unknown, snapshot is earlier, selected evidence is missing, or supplied sources disagree |

`evidence_status` explains which evidence condition produced that status:
`issuance_metadata_unavailable`, `claim_only`, `outcome_subset_only`,
`context_unverified`, `earlier_snapshot`, `no_empirical_samples`,
`bucket_unavailable`, `conflicting_sources`, `below_activation_threshold`, or
`matching_bucket_statistics`.

Malformed contracts, invalid numbers, duplicate records, declared cross-session
evidence, and future evidence raise `ValueError`. They are not silently converted
into an assessment of poor calibration. The bounded vocabulary does not include
`well_calibrated`, `correct`, or `physically_true` as assessment conclusions.

Threshold activation is not scientific sample sufficiency. Even an `assessable`
result has not measured calibration performance, statistical confidence, accuracy,
generalization, or Experiment 001 validity. Both entirely correct and entirely
incorrect historical outcomes can satisfy this evidence-availability policy.

## Empirical evidence and outcome eligibility

Statistics counts must be nonnegative integers. Their total must equal the sum
of correct, partial, and incorrect counts. Weighted success must equal the counts
under the existing producer weights. Weighted success is not restricted to one.
Reliability is copied, validated in `[0,1]`, and compared to the supplied issuance
value when it refers to an empirical selected bucket. It is not recomputed.
Agreement between two source outputs is not independent verification of either.

Only `correct`, `partially_correct`, and `incorrect` results enter the descriptive
evaluable outcome counts. Partial outcomes retain their own count and weight `0.5`.
`unevaluable` and `expired` have separate excluded counts. Missing outcomes are
not manufactured; an explicitly empty subset has zero supplied results, not zero
errors across all predictions. Unsupported statuses such as `pending` and
`unavailable` cannot masquerade as finalized public evaluation outputs.

The outcome adapter validates result/prediction identity, source/target/evaluation
time ordering, finite measurement fields, and original evaluation reasons.
Evaluable results require explicit observed same-track evidence and measurement
fields. It does not recompute geometric correctness or promote a status to truth.
Finalization remains explicitly caller-attested; callers must not label an early
`compare_prediction()` return as a finalized memory result.

Supplied outcome evidence has coverage
`supplied_subset_not_memory_membership`. Counts are derived only for the named
bucket. Outside-bucket outcomes and non-evaluable outcomes remain traceable by ID.
The summary records its derivation and original outcome weights.

Outcome subsets are never added to bucket totals or used to prove calibration
memory membership. Statistics snapshots do not list contributing prediction IDs,
and bucket retention is not a per-outcome sliding window: active bucket totals
can outlive individual retained result records. This layer neither replays memory
nor infers which supplied outcomes remain represented in its totals. An outcome
subset alone therefore does not corroborate an issuance reliability claim.

## Bucket specificity

The actual source/type/class prefix from `calibration_bucket` is preserved.
Class `None` remains explicitly unknown. A source fallback is not relabeled as
class-specific evidence. No unrelated buckets are substituted, combined, or
averaged. Other rows remain in the supplied snapshot for inspection.

For `global_prior`, the reported sample count concerns the source-level bucket.
If that row is supplied, it is preserved separately. Its empirical reliability
need not equal the issuance fixed prior: for example one correct outcome yields
source reliability `0.6`, while issuance still carries the fixed `0.5` prior.
These values remain distinct. No global AISTHESIS reliability is constructed.

## Temporal causality

Issuance context time means actual issuance/calibration capture time. It is not
inferred from `prediction_created_timestamp`: delayed/occlusion predictions may
have an older forecast origin, and the existing producer can calibrate at a later
state or memory time. The optional original creation time remains separate.

Known snapshot capture/update times and outcome evaluation times must not exceed
issuance. Each evidence wrapper also rejects source times later than its own known
capture time. An issued prediction cannot be its own historical evidence.

An earlier statistics snapshot is preserved but yields `indeterminate`, even when
its numeric values match: its retention/updates cannot establish the state at
issuance. Missing session/time context is explicitly unverified and prevents a
positive assessment from supplied evidence. Such source data remains inspectable;
it is not promoted to temporally valid evidence.

Empirical calibration is session-scoped and can span scenes. Optional scene/frame
labels describe caller capture context, not a claim that every historical outcome
shares one image. Declared session conflicts are rejected. No interpolation,
retrospective assessment, retrospective confidence rewrite, TTL update, or future
outcome feedback is implemented.

## Provenance and determinism

Every assessment preserves prediction identity/source/type/class, original
confidence fields, actual bucket, sample count, caller context, optional original
creation time, policy/version, source references and assumptions. Nested evidence
retains raw source records, capture times, last updates, original outcome statuses,
measurement fields, and caller provenance.

Issuance values and selected statistics are copied without numeric transformation.
Descriptive outcome counts explicitly state their derivation and weights. No
posterior, uncertainty, or physical-truth value is derived.

IDs use the repository's canonical content hashing. Statistics sort by bucket;
outcomes sort by content ID. Duplicate bucket rows or prediction outcome IDs are
rejected. Assumptions, reasons and references have canonical ordering. Serialization
detaches data and round-trips through ordinary JSON; no generic deserializer is
provided. Reconstructing records reruns their validations and derived statuses.

The existing bounded metadata utility limits depth, nodes, strings, keys and
serialized size (1 MiB per traversal), and rejects nonfinite/opaque/buffer data.
There is no UUID, wall clock, randomness, I/O, network, model download, GPU,
external framework, or new package dependency. No calibration-memory method is
invoked by the assessment layer.

## Step 27A and Bayesian boundaries

Step 27A remains frozen. Its existing calibration adapter can independently
represent the original `calibrate()` output. Step 27B does not force its assessment
into a Step 27A confidence field or require a change to those records. No additional
integration adapter is needed for v0.1.

Bayesian posterior is a distribution over caller-defined branch labels, which
may overlap or fail to exhaust physical events. No posterior is admitted as a
calibration claim. Explicit task-specific ground-truth semantics would be needed
for any future calibration study of that distribution.

## Mandatory distinctions and non-goals

```text
CONFIDENCE != PROBABILITY
CONFIDENCE != CORRECTNESS
CONFIDENCE != PHYSICAL TRUTH
CALIBRATED CONFIDENCE != GUARANTEE
CALIBRATION RELIABILITY != TRUTH PROBABILITY
CALIBRATION RELIABILITY != POSTERIOR
BAYESIAN POSTERIOR != CALIBRATED PROBABILITY
HIGH POSTERIOR != CORRECT
HIGH CONFIDENCE != CORRECT
LOW CONFIDENCE != INCORRECT
GOOD CALIBRATION != HIGH ACCURACY
HIGH ACCURACY != GOOD CALIBRATION
INSUFFICIENT CALIBRATION EVIDENCE != BAD CALIBRATION
NO CALIBRATION EVIDENCE != MIS-CALIBRATION
UNEVALUABLE OUTCOME != INCORRECT OUTCOME
MISSING OUTCOME != INCORRECT OUTCOME
PARTIALLY CORRECT != AUTOMATICALLY CORRECT
HISTORICAL RELIABILITY != CURRENT PHYSICAL FACT
SOURCE BUCKET RELIABILITY != GLOBAL SYSTEM RELIABILITY
EMPIRICAL FREQUENCY != PHYSICAL TRUTH PROBABILITY
GROUNDING SCORE != CALIBRATED CONFIDENCE
ROUTING RELEVANCE != CALIBRATION
EXPERIENCE SIMILARITY != CALIBRATION
UNCERTAINTY != 1 - CONFIDENCE
CALIBRATION ASSESSMENT != CALIBRATION ALGORITHM
```

No `1-confidence` or `1-reliability` conversion, global calibration score,
uncertainty pooling, branch selection, ECE, maximum calibration error, Brier score,
NLL, reliability diagram, calibration curve, bin optimization, temperature scaling,
Platt scaling, isotonic regression, model fitting, or recalibration is included.

## Validation

Dedicated suite: `evaluation/tests/test_calibration_assessment.py`.
It exercises real `PredictionEvaluationMemory` results and
`ConfidenceCalibrationMemory` outputs using synthetic inputs. Coverage includes
all status/bucket boundaries, partial/unevaluable/expired handling, immutable source
preservation, context and future-leakage checks, count consistency, finite/range
validation, provenance, deterministic IDs/ordering/serialization across hash seeds,
and rejection of Bayesian/grounding/uncertainty objects as calibration inputs.

Runs completed in the requested order:

| Suite | Result |
| --- | --- |
| Step 27B dedicated | 118 passed |
| Existing calibration | 17 passed, 2 subtests passed |
| Prediction evaluation | 12 passed, 3 subtests passed |
| Frozen Step 27A | 143 passed |
| Bayesian belief and belief prediction bridge | 220 passed |
| Relevant world-model regressions | 1,223 passed in 73.24 seconds |
| Full automated repository suite | 2,410 passed, 130 subtests passed in 101.58 seconds |

No existing tests have been changed, weakened, skipped or marked xfail.

Commands for the first five stages:

```powershell
.venv/Scripts/python.exe -m pytest -q evaluation/tests/test_calibration_assessment.py
.venv/Scripts/python.exe -m pytest -q test_confidence_calibration.py
.venv/Scripts/python.exe -m pytest -q test_prediction_evaluation.py
.venv/Scripts/python.exe -m pytest -q evaluation/tests/test_uncertainty_representation.py
.venv/Scripts/python.exe -m pytest -q -x evaluation/tests/test_bayesian_belief_state.py evaluation/tests/test_belief_prediction_bridge.py
```

Relevant world-model regression command:

```powershell
.venv/Scripts/python.exe -m pytest -q -x evaluation/tests/test_physical_world_model.py evaluation/tests/test_physics_constraints.py evaluation/tests/test_latent_physical_state.py evaluation/tests/test_cross_modal_consequences.py evaluation/tests/test_multiple_futures.py evaluation/tests/test_common_evidence_state.py evaluation/tests/test_reasoning_connectome.py evaluation/tests/test_reasoning_connectome_v02.py evaluation/tests/test_active_perception.py evaluation/tests/test_spatial_grounding.py evaluation/tests/test_locateanything_backend.py evaluation/tests/test_experience.py evaluation/tests/test_experience_learning.py evaluation/tests/test_experience_learning_v02.py evaluation/tests/test_astra_efa_experience.py evaluation/tests/test_hybrid_world_state.py evaluation/tests/test_complex_systems.py evaluation/tests/test_topological_world.py evaluation/tests/test_sensory.py evaluation/tests/test_reasoner_orchestration.py
```

The full automated run includes all 49 tracked test-case modules plus Step 27B,
with no test-case exclusions. As at the Step 27A freeze, the 13 legacy manual
scripts with no test cases are not imported; some immediately launch GPU/model
inference. Unrelated generated/untracked scripts are not executed.

```powershell
@'
import ast
from pathlib import Path
import subprocess
import pytest
suites = []
for name in subprocess.check_output(['git', 'ls-files', '*test*.py'], text=True).splitlines():
    path = Path(name)
    if not path.name.startswith('test_'):
        continue
    tree = ast.parse(path.read_bytes())
    if any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith('test_')
           for node in ast.walk(tree)):
        suites.append(name)
new_suite = 'evaluation/tests/test_calibration_assessment.py'
if new_suite not in suites:
    suites.append(new_suite)
raise SystemExit(pytest.main(['-q', '-x', *suites]))
'@ | .venv/Scripts/python.exe -
```

## Limitations, Step 27C and Experiment 001

This library is opt-in. No application/UI wiring is added. Adapters intentionally
accept only the inspected current public source shapes. Malformed or future
versions require an explicit extension, not silent coercion. Caller attribution
cannot establish source authenticity, atomic capture, independent evidence,
finalization, calibration-memory membership, or statistical representativeness.

The layer validates source structure and evidence availability, not the producer's
calibration formula through a second implementation. It does not assess current
prediction correctness or fit empirical calibration. Count sufficiency does not
establish reliable estimates for an experiment or generalization to new scenes.

Step 27C will specify appropriate calibration evaluation metrics and their required
data semantics. Experiment 001 will supply actual task-level comparisons between
baselines, AISTHESIS and ablations. Its question is whether reconstructing expected
but unavailable sensory consequences improves latent-state and future-event
inference under partial observability. Step 27B does not answer that question.

**No empirical calibration-performance, improved-prediction, or hidden-actor
anticipation claim follows from these engineering tests.**
