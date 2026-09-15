# Step 27A — Uncertainty Representation v0.1

## Status

🧩 IMPLEMENTED

🧪 TESTED — 143 dedicated tests passed; validation details below

📊 NOT YET TASK-LEVEL EVALUATED

🏆 NOT YET EXPERIMENTALLY SUPPORTED

Step 27A is an engineering representation layer. It is additive and opt-in.
It does not change any frozen producer, the application, or backend execution.
Step 26 remains closed; the real LocateAnything smoke test remains deferred.
This milestone has not been committed, tagged, or frozen.

## Purpose and inspection findings

AISTHESIS is a physics-constrained cross-modal predictive world model for
reasoning under partial observability. Its uncertainty representation answers:

> What uncertainty is represented here, about what, according to which source,
> under which assumptions, and with what evidence?

Inspection covered `confidence_calibration.py`, `prediction_evaluation.py`,
the world-model contracts and documentation, tracking/perception uncertainty,
both reasoning connectomes, Common Evidence State, active perception, spatial
grounding, and the LocateAnything bridge. Searches covered uncertainty,
confidence, posterior, probability, reliability, calibration, ambiguity,
indeterminate, and unavailable.

Existing meanings differ:

- Tracking position uncertainty can use pixels and is not bounded by one.
- Physical and latent states retain missing attributes, categorical limitations,
  constraint findings, and field references.
- Cross-modal consequences are conditional candidates, not sensor observations.
- Multiple Futures retains assumptions, qualitative horizons, and branch relations.
- Bayesian Belief State normalizes a categorical distribution over branch labels.
- `PredictionRecord` has separate optional `[0,1]` confidence and scalar uncertainty.
- Prediction evaluation measures outcome error; calibration uses finalized
  evaluable outcomes and a minimum of five samples. Neither establishes truth.
- Common Evidence State inventories uncertainty at original source paths.
- Reasoning connectomes retain source uncertainty and routing limitations;
  routing relevance is not probability.
- Active Perception proposes advisory reobservation targets from represented cues.
- Spatial Grounding retains unverified backend scores, labels, and candidate
  multiplicity. The LocateAnything bridge hands off through this frozen contract.

No common numeric scale is justified by these contracts.

## Architecture and API

```text
Explicit existing producer outputs
    -> source-specific adapters
    -> immutable UncertaintyEntry records
    -> UncertaintyRepresentationBuilder / UncertaintyBundle
    -> deterministic inspection and serialization
```

Production: `fifth_layer/world_model/uncertainty_representation.py`.
Adapters return tuples of entries. The builder takes an explicit context and
an ordered sequence of entries; it does not discover sources, execute producers,
query calibration memory, parse arbitrary producer dictionaries, or infer values.

```python
from fifth_layer.world_model.uncertainty_representation import (
    UncertaintyContext, UncertaintyRepresentationBuilder,
    adapt_prediction_uncertainty, bundle_from_dict,
)

# prediction is an existing PredictionRecord supplied by the caller.
entries = adapt_prediction_uncertainty(prediction)
context = UncertaintyContext(prediction.scene_id, prediction.source_timestamp)
bundle = UncertaintyRepresentationBuilder().build(context, entries)
assert bundle_from_dict(bundle.to_dict()) == bundle
```

## Data contracts

All public records are frozen dataclasses. Nested metadata is detached and
recursively immutable. `to_dict()` returns detached JSON-compatible data;
`to_json()` uses canonical key ordering and rejects nonfinite numbers.

### UncertaintyContext

`scene_id`, optional `timestamp`, optional `session_id`, and optional
`coordinate_frame_id`. Unknown values remain `None`. Timestamps are nonnegative
source times, never generated wall-clock values.

### UncertaintyEntry

| Field | Meaning |
| --- | --- |
| `uncertainty_id` | Content-derived deterministic identity |
| `context` | Original scene, source time, session, coordinate frame |
| `source_layer`, `source_type`, `source_id` | Originating module, public contract, source identity |
| `uncertainty_kind` | Explicit semantic category |
| `semantic_scope` | What the source field concerns |
| `status` | Availability or ambiguity of this representation |
| `value`, `value_kind`, `unit` | Copied value and its semantics; units only when explicitly supplied |
| `original_field`, `source_schema` | Original field path and schema, when available |
| `assumptions`, `source_references` | Canonical source assumption IDs and references |
| `provenance` | Source provenance and contextual source details |
| `transformation` | Always `none`; v0.1 supports copied values only |
| `schema_version`, `rule_version` | Fixed `uncertainty-representation-v0.1` |

Value kinds distinguish generic numeric, categorical, structured, confidence,
scalar uncertainty, branch-label distribution, backend score, and calibration
reliability. Only the source-defined confidence, scalar uncertainty, reliability,
and distribution probabilities enforce `[0,1]`. Generic numeric and backend-score
values require finiteness without imposing a probability range. Structured values
preserve source paths without guessing their scales from field names.

Categorical and structured values do not require manufactured numbers. Empty
explicit collections describe the supplied inventory, not zero world uncertainty.

### UncertaintyBundle

Contains `bundle_id`, `context`, a tuple of entries, source lineage, provenance,
and fixed schema/rule versions. Entries sort by content ID. Lineage consists of
sorted `(source_layer, source_type, source_id)` tuples. Exact duplicate entries
are rejected. Distinct sources or distinct field representations remain separate;
there is no score pooling or claim of independent evidence.

## Categories and explicit adapters

Categories are semantic labels, not scales:

| Category | Meaning and current adapter coverage |
| --- | --- |
| `observational` | Explicitly observed Common Evidence items, retaining original uncertainty paths and confidence |
| `spatial` | Grounded target ambiguity, regions, and backend score metadata |
| `temporal` | Multiple Futures qualitative horizon with unavailable numeric duration |
| `physical` | PhysicalWorldState, PhysicsConstraintBundle, LatentPhysicalState |
| `cross_modal` | CrossModalConsequenceBundle and its conditional candidates |
| `hypothesis` | MultipleFutureBundle branches, relations, assumptions, limitations |
| `belief` | BayesianBeliefState full branch-label posterior distribution |
| `prediction` | PredictionRecord confidence and uncertainty as separate entries |
| `calibration` | Exact explicitly supplied ConfidenceCalibrationMemory.calibrate output |
| `structural` | Common Evidence availability/integration, active-perception cues, reasoning snapshot uncertainty |

Public adapters are `adapt_physical_uncertainty`, `adapt_constraint_uncertainty`,
`adapt_latent_uncertainty`, `adapt_cross_modal_uncertainty`,
`adapt_future_uncertainty`, `adapt_belief_uncertainty`,
`adapt_prediction_uncertainty`, `adapt_calibration_uncertainty`,
`adapt_grounding_uncertainty`, `adapt_active_perception_uncertainty`,
`adapt_common_evidence_uncertainty`, and `adapt_reasoning_uncertainty`.

They accept the exact current contract classes and supported schemas. The
reasoning adapter accepts `ReasoningGraphSnapshotV02`. Calibration is the sole
mapping-shaped producer adapter because its current public output is a dict;
it requires exactly the five existing output fields and validates their shape.
No arbitrary-dictionary producer dispatch is provided.

Physical and latent adapters retain nested source records as structured details.
Common Evidence entries with epistemic status other than `observed` use the
structural category rather than implying actual observation. Active-perception
cue uncertainty remains structural; priority does not become an uncertainty score.

## Status and missing-information semantics

| Status | Representation meaning |
| --- | --- |
| `available` | An explicit field value is supplied; no correctness claim |
| `unavailable` | No usable value; `value=None` |
| `indeterminate` | Source ambiguity; may retain categorical/structured diagnostics or `None` |
| `insufficient_evidence` | Calibration sample threshold unmet; numeric source prior remains labeled as such |
| `not_applicable` | Explicit source not-applicable status; `value=None` |

Source `unsupported` maps to unavailable, `unknown` to indeterminate, and
`invalid_evidence` to unavailable posterior. Original source statuses and
diagnostics remain in provenance. A violated physical constraint remains a
source finding; it is not converted into a probability or relabeled as error.

- No grounding candidate means unavailable localization, not physical absence.
- An unavailable consequence does not mean that no sound or force exists.
- An unavailable posterior stays `None`, not zero or a uniform distribution.
- No constraint results means unavailable constraint information, not certainty.
- Unknown timestamp, session, and coordinate frame stay unknown.
- Absent adapters/sources are not fabricated by the builder. Callers can inspect
  bundle lineage; explicit Common Evidence availability covers its own families.

## Context and provenance policy

Declared scene/session/frame conflicts are rejected. Entry source timestamps
must be at or before the bundle cutoff. Earlier sources retain their own times;
no temporal alignment or interpolation is performed. A known source time cannot
be admitted against an unknown cutoff. Unknown source time stays `None`.

Missing session/frame context is permitted and remains visible. Even if the
bundle leaves either field unspecified, conflicting known source identities are
rejected. Direct conflicting identity declarations in source provenance are
rejected. Nested historical lineage retains its original context; it is not
reinterpreted as a current snapshot.

Entries preserve originating module/type, source schema when present, original
field name, source IDs/references, source provenance, assumption IDs and full
assumption records where available. Source details preserve future target time,
qualitative horizon, branch status, backend frame/region metadata, and other
source-specific diagnostics without promoting their meaning.

Physical/constraint inventories without a native bundle ID use their scene
snapshot identity with source time. Cross-modal/future bundles and latent records
without native IDs use explicitly labeled content references. Such references
are deterministic fingerprints, not invented upstream IDs.

Calibration has no native scene/time/source ID in its output. Its adapter requires
caller-supplied `context` and `source_id`, marks the context origin, and never
claims to have independently verified that attribution.

## Bayesian, calibration, and grounding boundaries

The belief adapter copies every posterior keyed by its hypothesis ID and
preserves full belief records, assumptions, and likelihood-evidence lineage.
It validates the supplied distribution sum without renormalizing it. Branch labels
can overlap and need not exhaust real-world events. Its scope remains
`categorical_branch_labels_not_event_frequencies`. There is no entropy, maximum
selection, winner, posterior-to-confidence conversion, or truth probability.

Calibration preserves reliability, sample count, bucket, raw confidence, and
calibrated confidence from an explicitly supplied issuance output. Below five
samples the `global_prior` output receives `insufficient_evidence`. Its numeric
reliability of 0.5 is retained as source metadata/value with that status, never
interpreted as 0.5 truth. No memory update, cleanup, or new calibration occurs.
Confidence and calibration reliability remain distinct semantic fields.

Grounding preserves all candidates and linked backend metadata. Each backend
score has its own entry and original observation ID, with no imposed `[0,1]`
range. Scores are not normalized, ranked, calibrated, or used to select a region.
LocateAnything provenance reaches this layer through the existing spatial
grounding output; this layer does not call its backend or parse raw responses.

## Mandatory semantic distinctions

```text
UNCERTAINTY != ERROR
UNCERTAINTY != 1 - CONFIDENCE
CONFIDENCE != PROBABILITY
CONFIDENCE != CALIBRATION
BAYESIAN POSTERIOR != CALIBRATED PROBABILITY
CALIBRATION RELIABILITY != PHYSICAL TRUTH
HIGH CONFIDENCE != CORRECT
LOW UNCERTAINTY != CORRECT
HIGH POSTERIOR != REALITY
HIGHEST POSTERIOR != WINNER/TRUTH
MISSING EVIDENCE != NEGATIVE EVIDENCE
MISSING SENSOR INFORMATION != ZERO UNCERTAINTY
INSUFFICIENT CALIBRATION DATA != 0.5 TRUTH
AMBIGUITY != RANDOMNESS
DISAGREEMENT != ERROR
SOURCE UNCERTAINTIES MUST NOT BE NAIVELY AVERAGED
GROUNDING SCORE != AISTHESIS CONFIDENCE
BACKEND SCORE != POSTERIOR
BACKEND SCORE != TRUTH PROBABILITY
ROUTING RELEVANCE != TRUTH PROBABILITY
EXPERIENCE SIMILARITY != PROBABILITY
TOPOLOGICAL VARIABILITY != ENTROPY
VARIABILITY != CHAOS
PREDICTION != OBSERVATION
POSSIBILITY != FACT
LATENT HYPOTHESIS != REALITY
```

## Aggregation non-goals and determinism

There is no mean, weighted average, global uncertainty, risk score, winner
confidence, `1-confidence`, `1-posterior`, entropy, or uncertainty-reduction engine.
Consumer aggregation would require a separately specified model and empirical
justification. Copied legacy aggregate fields, if present in structured source
metadata, retain their original field paths and meanings only.

IDs use the existing SHA-256 `stable_id` utility with canonical content.
Reference/assumption sets and bundle entries have deterministic ordering.
`bundle_from_dict` accepts only this exact serialized schema and verifies IDs,
versions, ordering, and lineage. This is round-trip reconstruction, not producer
dictionary parsing. No wall clock, UUID, randomness, I/O, network, download, GPU,
external framework, or feedback side effect is introduced.

The existing bounded metadata utility enforces finite numbers, depth 24, 50,000
visited nodes, 16,384-character strings, bounded keys, and a 1 MiB serialized
limit per traversal. Buffer/opaque payloads are rejected. Oversized entries or
bundles fail explicitly rather than truncating provenance.

## Validation

Dedicated tests: `evaluation/tests/test_uncertainty_representation.py`.

```powershell
.venv/Scripts/python.exe -m pytest evaluation/tests/test_uncertainty_representation.py -q
```

**143 passed**. Tests use real current producer objects and builders, including
synthetic calibration outcomes and grounding observations. Coverage includes
deterministic IDs/order/serialization across hash seeds, round trips and tamper
rejection, immutable detached data, numeric and categorical values, missing and
indeterminate states, all context conflicts, nonfinite rejection, source-specific
bounds, full posterior retention, calibration insufficiency, backend score
semantics, source permutation, mixed layers, provenance, assumptions, unchanged
sources, and import/I/O boundaries.

Relevant regression command:

```powershell
.venv/Scripts/python.exe -m pytest -q -x evaluation/tests/test_physical_world_model.py evaluation/tests/test_physics_constraints.py evaluation/tests/test_latent_physical_state.py evaluation/tests/test_cross_modal_consequences.py evaluation/tests/test_multiple_futures.py evaluation/tests/test_bayesian_belief_state.py evaluation/tests/test_belief_prediction_bridge.py evaluation/tests/test_common_evidence_state.py evaluation/tests/test_reasoning_connectome.py evaluation/tests/test_reasoning_connectome_v02.py evaluation/tests/test_active_perception.py evaluation/tests/test_spatial_grounding.py evaluation/tests/test_locateanything_backend.py test_confidence_calibration.py test_prediction_evaluation.py
```

The initial run stopped with **1,060 passed; 2 failed in 68.65 seconds**.
Both failures were subtests (`count=0`
and `count=5`) of
`test_confidence_calibration.py::CalibrationTests::test_actual_overlay_collecting_and_calibrated`.
At line 191 it calls `live_functions('draw_overlay')`. The existing helper at
`test_analysis_connections.py:21` executes
`ast.parse(Path("live_app.py").read_text())`, which fails at line 1 with
`SyntaxError: invalid character '\xbb' (U+00BB)` under the local cp1252 default.

Read-only baseline diagnosis confirmed that committed `HEAD:live_app.py` starts
with the UTF-8 BOM bytes `ef bb bf`. Decoding that committed file with the current
default encoding and parsing it reproduces the same error without importing
Step 27A. The application and test helper match HEAD after line-ending
normalization; calibration production/test files match HEAD byte-for-byte.
This is an existing local encoding/test-helper issue, independently reproducible
without importing Step 27A. No producer or application was changed.

The separately authorized infrastructure fix changes only
`test_analysis_connections.py:21` from `read_text()` to `read_bytes()`.
`ast.parse` then applies Python source-encoding semantics to the bytes, including
UTF-8 BOM and encoding-cookie handling. This avoids the cp1252 default and also
supports Python's default UTF-8 source encoding. Both current and committed HEAD
source bytes parse successfully. `live_app.py`, calibration semantics, and Step
27A production/test files remain unchanged by the fix.

Validation resumed in the requested order:

| Check | Result |
| --- | --- |
| Previously failing calibration test | 1 passed, 2 subtests passed |
| `test_confidence_calibration.py` | 17 passed, 2 subtests passed |
| Step 27A dedicated | 143 passed |
| Same relevant regression set above | 1,088 passed, 5 subtests passed in 62.20 seconds |
| Full automated repository suite | 2,292 passed, 130 subtests passed in 109.64 seconds |

No tests were weakened, deleted, skipped, or marked xfail.

An unrestricted root pytest discovery is unsuitable for this local run: legacy
manual scripts such as `test_yolo_detector.py`, `test_smolvlm_core.py`, and
`test_perception_fusion.py` immediately initialize model/GPU inference on import.
The full automated run selects all 48 tracked test-case modules plus the Step 27A
module. Selection inspects the AST of tracked `test_*.py` files for test functions
or methods without importing manual scripts. All test cases in those 49 modules
are included. The 13 legacy manual scripts contain no test cases and are not
modified or executed; unrelated untracked scripts are also untouched.
No commit or tag is authorized yet.

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
suites.append('evaluation/tests/test_uncertainty_representation.py')
raise SystemExit(pytest.main(['-q', '-x', *suites]))
'@ | .venv/Scripts/python.exe -
```

## Known limitations and later steps

- This is an opt-in library, with no live application integration.
- It does not independently verify producer truth, calibration, or provenance.
- Missing context limits alignment claims. Historical sources are not resampled.
- It preserves repeated lineage without establishing statistical independence.
- It does not add adapters for legacy live outcome dictionaries, calibration
  `statistics()` snapshots, raw backend responses, or older connectome snapshots.
  Outcome error remains in the existing prediction evaluation layer. Current
  calibration issuance outputs and Step 26B grounded results are supported.
- Nested source information stays structured where extracting a scalar would
  lose meaning. No category-specific aggregation is implemented.

Step 27B may specify empirical uncertainty/calibration evaluation and appropriate
data requirements. Step 27C may extend that separately specified work. Neither
stage is implemented or scientifically established by Step 27A.

Experiment 001 later asks whether reconstructing expected but unavailable sensory
consequences improves latent-state and future-event inference under partial
observability. Step 27A does not answer that question. It prepares inspectable
source information for predictive accuracy, calibration, uncertainty behavior,
time-to-anticipation, ablations, and baseline comparisons.

**Step 27A does not demonstrate that AISTHESIS uncertainty is well calibrated.**

**Step 27A does not demonstrate improved prediction.**

**Step 27A does not validate hidden-actor anticipation.**

Unit-test success is engineering validation, not scientific success.
