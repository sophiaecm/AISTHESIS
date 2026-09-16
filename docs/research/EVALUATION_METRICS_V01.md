# Evaluation Metrics v0.1

## Purpose and scientific boundary

Step 27C provides descriptive measurement infrastructure for explicitly supplied
prediction inventories and finalized geometric evaluation evidence. A metric
describes those supplied records; it does not establish physical truth or show
that AISTHESIS outperforms a baseline.

```text
CONFIDENCE != PROBABILITY
[0,1] VALUE != PROBABILITY
METRIC RESULT != PHYSICAL TRUTH
METRIC INELIGIBLE != MODEL FAILURE
MISSING METRIC != ZERO PERFORMANCE
UNEVALUABLE != INCORRECT
EXPIRED != INCORRECT
PENDING != INCORRECT
MISSING OUTCOME != INCORRECT
INSUFFICIENT DATA != POOR PERFORMANCE
HIGH SCORE != SCIENTIFIC VALIDATION
PARTIALLY_CORRECT != STRICTLY CORRECT
BAYESIAN POSTERIOR != AUTOMATICALLY A CALIBRATED FORECAST
CALIBRATED CONFIDENCE != GUARANTEE
CALIBRATION RELIABILITY != TRUTH PROBABILITY
GROUNDING SCORE != PROBABILITY
ROUTING RELEVANCE != PROBABILITY
EXPERIENCE SIMILARITY != PROBABILITY
UNCERTAINTY != 1 - CONFIDENCE
GOOD CALIBRATION != HIGH ACCURACY
HIGH ACCURACY != GOOD CALIBRATION
NO FUTURE LEAKAGE
```

## Architecture and input/evaluation contracts

`fifth_layer/world_model/evaluation_metrics.py` consumes detached evidence through
`adapt_finalized_evaluation` or `represent_unresolved_prediction`, producing frozen
`EvaluationUnit` records. `EvaluationMetricsBuilder.build` creates an
`EvaluationMetricsReport` with coverage, outcome distribution, metric results,
source references and provenance. There is no live application integration,
producer execution, fitting, calibration update or overall system score.

Each unit requires a typed `CalibrationIssuance`. Finalized input is the exact
current evaluation mapping or a `FinalizedCalibrationOutcome`; the adapter
requires `finalized=True` as an explicit caller attestation. It validates the
field set, original prediction lineage, source/type/class and calibration
metadata, timestamps, status, evaluation reason, measurements and observation
geometry. It compares the embedded prediction with the captured source prediction
when present. Arbitrary score dictionaries are not trusted outcomes.

Unresolved inventory explicitly distinguishes `pending` from `missing_outcome`.
It cannot carry a finalized outcome or reference. Duplicate prediction IDs are
rejected across the supplied report inventory. Report session conflicts and
known timestamps later than an optional report cutoff are rejected.

## Shared timing contract extraction

`EVALUATION_GRACE_SECONDS` remains **0.25 seconds**. Its single definition moved
from `fifth_layer/prediction_evaluation.py` to the import-free lightweight module
`fifth_layer/evaluation_contracts.py`. The producer imports and continues to expose
the same public symbol. Step 27C imports the lightweight contract directly.

The authorized frozen-file change replaces only the producer's constant
definition with that import. No evaluation algorithm, timing, finalization,
observation matching or outcome semantics changed. This separates measurement
infrastructure from the producer's inherited perception/image imports.

## Metric eligibility

`MetricEligibility` exposes status, reasons, available sample count, source/target
semantics, requirements met/missing and policy version. Statuses are `eligible`,
`insufficient_data`, `ineligible` and `protocol_required`.

`MetricResult` exposes numerator, denominator, denominator definition, value,
sample count, contributing unit IDs, source references, interpretation and
transformation. Unavailable results have `value`, `numerator` and `denominator`
set to `None`, sample count zero, no contributing units and `not_computed` as
their transformation. Their source references remain inspectable.

## Coverage semantics and exclusion accounting

Coverage records total supplied records, eligible and excluded counts,
source-evaluable count, unqualified source-evaluable count, and separate
unevaluable, expired, pending and missing-outcome counts. Source-evaluable means
the source labeled an outcome correct, partially correct or incorrect; eligibility
additionally requires known issuance chronology and an explicit reference ID.

`evaluation_coverage` is eligible count divided by all supplied unique prediction
units. It describes only the supplied inventory, not all predictions ever issued.
An all-excluded nonempty inventory legitimately has coverage zero. An empty
inventory has unavailable coverage. Neither case fabricates zero performance.

Every excluded unit retains its source record, identity, provenance and sorted
exclusion reasons. Multiple reasons on one unit count as one excluded record.
Unknown issuance time and missing reference identity are explicit exclusions.
Malformed or contradictory input is rejected rather than inserted into a report;
callers must retain rejected input separately if they need an ingestion audit.

## Outcome distribution, weighted outcome score and strict accuracy

The denominator for every outcome metric is
`finalized_evaluable_units_with_reference_and_issuance_chronology`.
Let N be that count and C, P, I be original correct, partially-correct and
incorrect counts. `OutcomeDistribution` preserves all three counts and N.

| Metric | Definition |
| --- | --- |
| `correct_outcome_frequency` | C / N |
| `partial_outcome_frequency` | P / N |
| `incorrect_outcome_frequency` | I / N |
| `weighted_outcome_score` | (1.0 C + 0.5 P + 0.0 I) / N |
| `strict_accuracy` | C / N; only original `correct` is strict success |
| `incorrect_outcome_rate` | I / N; not one minus weighted outcome score |

Weights are imported from the existing confidence-calibration contract and
frozen locally. Weighted outcome score is explicitly weighted, not ordinary
accuracy or probability. Partial outcomes are never silently promoted to correct.
When N is zero, all outcome metric values are `None` with `insufficient_data`.
Excluded records never become incorrect outcomes.

## Temporal validity / no future leakage

The forecast origin cannot follow its known issuance/capture time. For evaluable
outcomes, both forecast origin and known issuance time must strictly precede the
reference observation. The observation must be no later than evaluation and
within 0.25 seconds of the target. Evaluation must occur at or after target plus
0.25 seconds, closing the producer finalization window. Evaluation may happen
later than the observation. Unknown issuance time excludes the unit from
performance metrics; known contradictory chronology raises `ValueError`.

The report cutoff constrains known issuance and evaluation times. No wall-clock
time is generated, no chronology is inferred from confidence, and no prediction
is rewritten. Caller attribution and finalization are validated structurally;
the library cannot independently prove when evidence was actually captured.

## Brier, NLL and ECE eligibility

All three remain `ineligible` with unavailable values. Current bounded confidence
and reliability fields do not supply contractual probability forecasts paired
with resolved targets for the same events. ECE cannot interpret engineering
confidence as an event probability merely because it lies in [0,1].

`probability_metric_eligibility` accepts typed calibration issuance, Bayesian
belief state or Multiple Futures sources to explain ineligibility. Raw numbers
and arbitrary dictionaries cannot declare themselves probability contracts.
There is no conversion from confidence, reliability, grounding, routing,
experience similarity, posterior or uncertainty to forecast probability.

## Bayesian posterior and Multiple Futures boundaries

Bayesian posterior scope is a categorical distribution over branch labels.
Branches need not be mutually exclusive or exhaustive real-world event classes;
normalization does not create a calibrated forecast with a resolved event target.
Multiple Futures represents conditional possibilities, not measured event labels.
Neither source is converted to geometric outcome accuracy. Their probability
eligibility records identify the missing event semantics explicitly.

## TTA / Experiment 001 protocol boundary

`tta` is `protocol_required` with value `None`. Repository review found no fixed
Experiment 001 Time-to-Anticipation protocol sufficient for computation. Missing
requirements are event onset, valid anticipation, anticipation threshold,
temporal alignment, false anticipation handling and censoring policy. Step 27C
does not invent these definitions. Experiment 001 has not been performed here.

## Provenance, determinism and immutability

Units retain issuance, source prediction IDs, source/reference IDs, caller
provenance and exclusion reasons. Reports retain dataset/session attribution,
optional cutoff, assumptions, all units and the reference union. Metric lineage
identifies its contributing units and references explicitly.

Public records are frozen dataclasses with detached, recursively frozen nested
metadata. `to_dict()` returns detached JSON-compatible content; `to_json()` uses
sorted keys, compact separators and rejects nonfinite values. Units and reports
have content-derived stable IDs using `evaluation-metrics-v0.1`; unit ordering,
metric ordering and reference sets are deterministic. IDs use neither UUIDs nor
wall-clock identity. Finite-number and bounded metadata validation reject NaN,
infinity, cycles, oversized and opaque payloads. No network, GPU or model work
is performed by the metric layer.

## Relationship to Step 27A and Step 27B

Step 27A preserves distinct uncertainty meanings without pooling them. Step 27C
does not consume uncertainty as probability or compute `1-confidence`.

Step 27B supplies immutable issuance and finalized outcome contracts and assesses
calibration evidence. Step 27C reuses issuance validation and existing outcome
weights, but measures descriptive outcomes separately from calibration
reliability. It does not mutate either calibration or prediction-evaluation state.

## Limitations and relationship to Experiment 001

Geometric observations and producer labels are reference evidence, not physical
truth. Inventory selection, missing context, caller attribution, dependent
samples and source errors limit interpretation. The layer does not re-evaluate
geometric labels, establish sample independence, provide confidence intervals,
test significance, rank systems or demonstrate generalization. Brier/NLL/ECE
require a future explicit probability-forecast contract; TTA requires an
Experiment 001 protocol. Baselines, ablations and actual task-level evaluation
remain future work.

## Validation

Use `.venv/Scripts/python.exe`; the system Python lacks pytest. Initial validation
after extraction: **111 dedicated tests passed**; prediction evaluation:
**12 passed, 3 subtests passed**. Fresh interpreters verified both constant import
paths equal 0.25 and that importing Step 27C loads none of `numpy`, `torch`, `cv2`,
`transformers`, `PIL`, or the prediction-evaluation producer.

The existing dedicated test was retained unchanged. It covers deterministic
serialization/IDs across hash seeds, immutable detached records, coverage,
denominators, exclusions, strict/weighted outcomes, unavailable metrics,
probability boundaries, temporal rejection, producer-state preservation,
nonfinite rejection and import boundaries. Full automated discovery follows the Step 27B AST-based
selection of tracked test-case modules and includes the new Step 27C test file;
legacy manual scripts with no test cases and unrelated untracked scripts are
not imported.

Ordered validation on 2026-09-16:

| Check | Result |
| --- | --- |
| Step 27C dedicated, repeated after documentation | 111 passed |
| Prediction evaluation | 12 passed, 3 subtests passed |
| Confidence calibration | 17 passed, 2 subtests passed |
| Step 27B calibration assessment | 118 passed |
| Step 27A uncertainty representation | 143 passed |
| Bayesian Belief State and belief-prediction bridge | 220 passed |
| Multiple Futures | 144 passed |
| Relevant world-model set from Step 27B | 1,223 passed |
| Full automated repository suite, 51 modules | 2,521 passed, 130 subtests passed in 43.55 seconds |

The prior full-suite reference was 2,410 passed plus 130 subtests. The actual
new run adds the 111 existing Step 27C tests; no tests were weakened, removed,
skipped or marked xfail to obtain these results.

Individual commands use `.venv/Scripts/python.exe -m pytest -q -x` followed by,
in order: `evaluation/tests/test_evaluation_metrics.py`,
`test_prediction_evaluation.py`, `test_confidence_calibration.py`,
`evaluation/tests/test_calibration_assessment.py`,
`evaluation/tests/test_uncertainty_representation.py`, the pair
`evaluation/tests/test_bayesian_belief_state.py evaluation/tests/test_belief_prediction_bridge.py`,
and `evaluation/tests/test_multiple_futures.py`.
The relevant regression command is the exact 20-module world-model set documented
in [Step 27B](CALIBRATION_ASSESSMENT_V01.md).

Full automated command (PowerShell):

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
new_suite = 'evaluation/tests/test_evaluation_metrics.py'
if new_suite not in suites:
    suites.append(new_suite)
print('Automated test modules:', len(suites), flush=True)
raise SystemExit(pytest.main(['-q', '-x', *suites]))
'@ | .venv/Scripts/python.exe -
```

## Scientific status

Implemented and tested: dedicated, ordered regressions and full automated suite
passed. Not yet task-level evaluated. Not yet experimentally supported.
Test success is engineering validation, not scientific validation.
