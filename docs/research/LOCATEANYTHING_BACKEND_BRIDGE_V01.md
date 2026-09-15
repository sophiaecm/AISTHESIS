# LocateAnything Backend Bridge v0.1

## Status

**Step 26C — LocateAnything Backend Bridge v0.1**

- 🧩 IMPLEMENTED
- 🧪 DEDICATED TESTED — 68 tests passed
- 🧪 REGRESSION TESTED — completed with 0 errors
- 📊 REAL LOCATEANYTHING BACKEND NOT YET SMOKE-TESTED
- 📊 NOT YET TASK-LEVEL EVALUATED
- 🏆 NOT YET EXPERIMENTALLY SUPPORTED

Passing engineering tests establish implementation behavior only. They do not establish grounding accuracy, task benefit, physical-world correctness, or scientific support.

## Purpose

Step 26C defines an isolated backend boundary between AISTHESIS Step 26B Spatial Grounding and an external LocateAnything-compatible runtime.

The bridge is intentionally not a model loader. It does not download, initialize, or execute LocateAnything by itself. Runtime execution is injected through a narrow interface so that external model output can be validated and converted into the frozen Step 26B `RawGroundingObservation` contract without allowing the backend to mutate AISTHESIS world state.

Conceptually:

```text
Step 26A Active Perception
        |
        v
ObservationRequestPlan
        |
        v
Step 26B Spatial Grounding
        |
        v
GroundingRequest
        |
        v
Step 26C LocateAnything Backend Bridge
        |
        +--> LocateAnythingInvocation
        |
        +--> injected external runtime
        |
        +--> LocateAnythingRawResponse
        |
        +--> strict parser / lineage validation
        |
        v
RawGroundingObservation(s)
        |
        v
Step 26B SpatialGroundingAdapter
```

## Scope

This version provides:

- an immutable LocateAnything invocation record;
- an immutable raw backend-response record;
- an injected runtime protocol;
- exact request / invocation / response lineage checks;
- explicit caller-provided query handling;
- narrow parsing for box and point outputs;
- structured response parsing;
- normalized `0..1000` coordinate handoff to Step 26B;
- preservation of multiple candidates;
- explicit no-candidate handling;
- backend metadata and uncertainty preservation;
- deterministic content-derived identities;
- explicit epistemic safety provenance.

This version does **not** provide:

- model download or model loading;
- network access;
- Hugging Face execution;
- EAGLE worker execution;
- GPU compatibility validation;
- real image inference;
- real visual-grounding validation;
- camera control;
- robot control;
- hidden-actor detection;
- metric 3D localization;
- Bayesian updates;
- ExperienceLearning feedback;
- future-branch selection;
- world-state mutation;
- task-level performance claims.

## Backend Identity

The bridge declares the external backend boundary as:

```text
backend_name  = LocateAnything
backend_model = nvidia/LocateAnything-3B
schema        = locateanything-backend-v0.1
parser        = explicit-normalized-box-point-v0.1
```

These identifiers describe the intended external interface. They do not prove that the real model has been installed, loaded, executed, or validated on the current machine.

## Query Boundary

Step 26B intentionally leaves semantic query resolution unavailable.

Step 26C therefore does not infer a natural-language query from object IDs, relation IDs, labels, or world-model state. A query must be supplied explicitly by the caller.

```text
symbolic target != semantic model query
object ID != verified semantic identity
caller query != physical fact
```

The invocation provenance records that the query was caller-provided.

## Runtime Boundary

The runtime is injected through a minimal interface equivalent to:

```text
infer(invocation) -> LocateAnythingRawResponse
```

The bridge validates the returned response against the invocation before parsing it.

A runtime response must preserve:

- invocation ID;
- request ID;
- frame ID;
- backend name;
- backend model.

A lineage mismatch is rejected rather than repaired.

## Output Grammar

The text parser intentionally accepts only a narrow explicit representation.

A four-coordinate record represents a box:

```text
<box><x1><y1><x2><y2></box>
```

A two-coordinate record inside the same upstream wrapper represents a point:

```text
<box><x><y></box>
```

An explicit no-candidate response is:

```text
<no_candidate/>
```

Arbitrary numbers appearing in prose are not interpreted as coordinates.

Structured responses are also supported through an explicit payload containing a status and result records. Structured records must identify their output type and coordinates. Unknown or malformed record fields are rejected.

If structured coordinates and text coordinates compete in the same response, the bridge rejects the ambiguous representation rather than selecting one.

## Coordinate Semantics

Step 26C emits backend observations in:

```text
normalized_0_1000
```

The bridge does not claim that these values are:

- metric world coordinates;
- depth;
- object centers;
- physical dimensions;
- 3D poses.

Step 26B remains responsible for validated conversion from normalized coordinates to the current frame's continuous pixel coordinates.

Coordinate values must satisfy the frozen Step 26B observation contract. Invalid arity, negative coordinates, values beyond the normalized boundary, reversed boxes, and non-finite numeric metadata are rejected.

Non-finite values such as NaN and infinity are rejected at the raw-response structured-data boundary before they can become grounding observations.

## Multiple Candidates

Multiple valid backend candidates remain multiple candidates.

The bridge does not:

- rank them into physical truth;
- select a winner;
- infer which candidate is the real object;
- promote backend scores into AISTHESIS confidence.

Step 26B can therefore preserve the result as `multiple_candidates`.

## Backend Scores and Labels

Backend labels and scores are metadata only.

A backend label is not verified object identity.

A backend score is not:

- AISTHESIS confidence;
- Bayesian posterior;
- probability of truth;
- risk;
- physical-state confidence.

Missing scores remain missing.

## No-Candidate Semantics

An explicit no-candidate backend result may be represented as an empty observation sequence and later integrated by Step 26B as `no_candidate`.

This means only that the backend supplied no localization candidate under the given request.

It does **not** mean:

```text
object absent
hidden actor absent
physical entity absent
world-state hypothesis false
```

Therefore:

```text
NO GROUNDING != PHYSICAL ABSENCE
```

## Epistemic Safety Rules

The following distinctions are mandatory in Step 26C:

```text
LOCATEANYTHING OUTPUT != PHYSICAL TRUTH

PARSED BOX != VERIFIED OBJECT

PARSED POINT != VERIFIED OBJECT CENTER

BACKEND TEXT != AISTHESIS OBSERVATION FACT

BACKEND LABEL != VERIFIED IDENTITY

BACKEND SCORE != AISTHESIS CONFIDENCE
BACKEND SCORE != BAYESIAN POSTERIOR
BACKEND SCORE != TRUTH PROBABILITY
BACKEND SCORE != RISK

GROUNDING SUCCESS != WORLD-MODEL CORRECTNESS

NO GROUNDING != PHYSICAL ABSENCE

MULTIPLE BOXES != MULTIPLE REAL OBJECTS

BOX COORDINATES != METRIC 3D POSITION

POINT != DEPTH

GROUNDING != HIDDEN ACTOR DETECTION

GROUNDING != CAMERA CONTROL
GROUNDING != ROBOT CONTROL

GROUNDING != BAYESIAN UPDATE

GROUNDING != EXPERIENCE LEARNING
```

The bridge records limitations including:

- `external_backend_output_unverified`
- `grounding_not_physical_truth`
- `backend_label_not_verified_identity`
- `backend_score_not_aisthesis_confidence`
- `no_candidate_not_physical_absence`
- `normalized_2d_not_metric_3d`
- `grounding_not_control`
- `real_backend_not_smoke_tested`

## Relationship to Active Perception

Step 26A produces observation targets and priorities.

Step 26B converts a selected symbolic target into a grounding request.

Step 26C provides an external-backend adapter for that request.

This does not yet make AISTHESIS an autonomous active-perception system. There is no sensor actuation or camera movement in Step 26C.

The current relationship is:

```text
reason about what may be useful to observe
        !=
physically execute an observation action
```

## Relationship to the World Model

Step 26C does not modify:

- `LatentPhysicalState`;
- `TopologicalWorldState`;
- `ComplexSystemsState`;
- Bayesian belief state;
- future branches;
- ASTRA-EFA experience state.

External grounding evidence remains downstream evidence until another explicitly designed and validated layer decides how it may be used.

This prevents a visual-grounding backend from silently becoming a source of physical truth.

## Testing

The dedicated Step 26C suite uses synthetic requests and an injected fake runtime.

It tests, among other things:

- immutability;
- deterministic identities;
- request preservation;
- explicit query requirements;
- output-type preservation;
- invocation and response lineage;
- runtime call isolation;
- text box parsing;
- text point parsing;
- coordinate boundaries;
- malformed-coordinate rejection;
- arbitrary-prose-number rejection;
- multiple-candidate preservation;
- structured box/point parsing;
- unrequested-output rejection;
- non-finite structured-data rejection;
- explicit no-candidate behavior;
- ambiguous dual-representation rejection;
- backend-score non-promotion;
- epistemic provenance;
- Step 26B integration;
- absence of model-loader, network, subprocess, and control behavior.

Observed validation result during Step 26C development:

```text
Dedicated Step 26C:
68 passed
0 failed

Regression:
completed with 0 errors
```

These results are engineering validation, not scientific evaluation.

## Current Scientific Status

Step 26C establishes an implementation boundary, not evidence that LocateAnything improves AISTHESIS.

At this stage we have not measured:

- grounding accuracy on real images;
- latency on the target hardware;
- GPU memory requirements;
- robustness under occlusion;
- calibration;
- downstream prediction improvement;
- hidden-state inference improvement;
- Time-to-Anticipation improvement;
- Experiment 001 performance.

Accordingly:

```text
🧩 IMPLEMENTED
🧪 TESTED
📊 REAL BACKEND NOT YET SMOKE-TESTED
📊 NOT YET TASK-LEVEL EVALUATED
🏆 NOT YET EXPERIMENTALLY SUPPORTED
```

## Next Step

The next engineering checkpoint is a separate real-backend smoke-test stage.

That stage should validate the actual external runtime without weakening the Step 26C boundary. It should begin with installation/runtime feasibility and a minimal controlled image test, then verify that real backend outputs can be represented through the same invocation -> response -> parser -> Step 26B path.

Real-backend success must remain distinct from scientific task success.

```text
real backend runs
    !=
grounding is accurate
    !=
world model is correct
    !=
prediction improves
    !=
Experiment 001 is supported
```
