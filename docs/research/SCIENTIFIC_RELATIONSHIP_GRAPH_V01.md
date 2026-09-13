# AISTHESIS Scientific Relationship Graph v0.1

## Status

🧩 IMPLEMENTED  
🧪 TESTED  
📊 NOT YET TASK-LEVEL EVALUATED  
🏆 NOT YET EXPERIMENTALLY SUPPORTED

---

## Purpose

Scientific Relationship Graph v0.1 is the third stage of the
AISTHESIS Mini-Lab scientific-document pipeline.

It converts scientific-structure candidates into an immutable,
provenance-preserving graph of candidate scientific relationships.

Pipeline:

Scientific PDF
→ ScientificDocumentState
→ ScientificStructureState
→ ScientificRelationshipGraph

This milestone represents possible relationships.

It does not verify that those relationships are scientifically true.

---

## Core Epistemic Boundary

The following distinctions are mandatory:

relationship candidate != verified scientific relationship

graph edge != scientific fact

same-page proximity != semantic relationship

supports edge != demonstrated scientific support

contradicts edge != demonstrated contradiction

supported_by edge != validated evidence support

structural association != causality

repeated relationship != stronger evidence

inference method != scientific justification

extractor confidence != truth probability

The graph represents candidate relations for later reasoning.

It does not decide scientific truth.

---

## Inputs

ScientificRelationshipBuilder consumes:

1. ScientificDocumentState
2. ScientificStructureState

Both inputs must refer to the same document.

A document identity mismatch is rejected.

The builder does not parse PDF files directly.

PDF ingestion remains the responsibility of:

Scientific Document Input Core v0.1

Scientific structure extraction remains the responsibility of:

Scientific Structure Model v0.1

---

## Output

The primary output is:

ScientificRelationshipGraph

containing immutable:

ScientificRelationshipCandidate

objects.

Each relationship preserves:

- relationship ID
- relationship type
- epistemic status
- source node reference
- target node reference
- document ID
- source element provenance
- optional confidence
- inference method

---

## Graph Nodes

Scientific Relationship Graph does not duplicate scientific
structures or document elements.

Instead, it uses:

ScientificGraphNodeRef

A graph node reference contains:

- node_id
- node_kind
- document_id

Supported node kinds are:

- STRUCTURE_CANDIDATE
- DOCUMENT_ELEMENT

This preserves identity across Mini-Lab stages.

Canonical provenance chain:

PDF
↓
DocumentElement
↓
ScientificStructureCandidate
↓
ScientificRelationshipCandidate
↓
future Scientific Evidence

---

## Relationship Types

Scientific Relationship Graph v0.1 represents:

- TESTED_BY
- USES
- MEASURES
- ABOUT
- SUPPORTED_BY
- DESCRIBES
- MODELS
- SUPPORTS
- CONTRADICTS
- UNKNOWN

Not every represented relationship type is automatically inferred
by the v0.1 builder.

The enum defines the graph vocabulary.

The builder uses only a conservative subset.

---

## v0.1 Structural Rules

ScientificRelationshipBuilder v0.1 currently supports the following
candidate mappings:

HYPOTHESIS
→ TESTED_BY
→ METHOD

METHOD
→ USES
→ DATASET

METHOD
→ MEASURES
→ VARIABLE

RESULT
→ ABOUT
→ VARIABLE

RESULT
→ SUPPORTS
→ HYPOTHESIS

CLAIM
→ SUPPORTED_BY
→ RESULT

These mappings are candidate-generation rules only.

For example:

RESULT → SUPPORTS → HYPOTHESIS

does not mean that AISTHESIS has established that the result
scientifically supports the hypothesis.

The edge records a heuristic candidate relationship for later
evaluation.

---

## Same-Page Constraint

v0.1 uses a deliberately narrow structural constraint.

A relationship is generated only when the source and target
structure candidates occur on the same human-readable page.

If either page is unknown, no relationship is generated.

If the structures occur on different pages, no relationship is
generated.

This constraint reduces unrestricted combinatorial linking.

However:

same page != semantic relationship

Two scientific structures occurring on the same page may be
unrelated.

Therefore all automatically generated relationships remain:

HEURISTIC

---

## Relationship Status

ScientificRelationshipStatus supports:

- CANDIDATE
- HEURISTIC
- UNKNOWN

The v0.1 builder emits:

HEURISTIC

for automatically inferred relationships.

The builder does not promote an inferred relationship to a verified
or factual status.

---

## Confidence

ScientificRelationshipCandidate supports optional confidence for
future compatibility.

The v0.1 builder assigns:

confidence = None

This is intentional.

The current structural rules do not provide a scientifically
meaningful calibrated probability.

No numerical truth confidence is fabricated from same-page
proximity.

Future models may expose inference confidence, but such confidence
must remain distinct from:

- scientific truth
- evidence strength
- hypothesis support
- causal confidence
- calibrated belief

---

## Provenance

Every relationship candidate must preserve source provenance.

source_element_ids contains the document elements associated with
both sides of the candidate relationship.

For example:

HypothesisCandidate
source: element-h

MethodCandidate
source: element-m

may produce:

Hypothesis
→ TESTED_BY
→ Method

with provenance:

element-h
element-m

Provenance is merged deterministically.

It is not discarded when graph edges are created.

---

## Identity

Relationship IDs are deterministic.

Identity includes:

- document identity
- relationship type
- source node kind
- source node ID
- target node kind
- target node ID

Given identical inputs, the builder is designed to produce
identical relationship identities.

---

## Directionality

Scientific relationships are directional.

For example:

HYPOTHESIS
→ TESTED_BY
→ METHOD

is not equivalent to:

METHOD
→ TESTED_BY
→ HYPOTHESIS

Similarly:

CLAIM
→ SUPPORTED_BY
→ RESULT

is distinct from:

RESULT
→ SUPPORTS
→ CLAIM

The graph preserves source and target direction explicitly.

---

## Self Relationships

Self relationships are rejected.

A graph node cannot create a relationship to itself when both
node identity and node kind are identical.

This prevents meaningless reflexive edges from entering the
candidate graph.

---

## Duplicate Relationships

Exact duplicate graph edges are suppressed.

Duplicate identity is defined using:

- relationship type
- source node kind
- source node ID
- target node kind
- target node ID

Duplicate suppression does not increase confidence.

Repeated generation of the same edge does not constitute stronger
scientific evidence.

---

## Determinism

Scientific Relationship Graph v0.1 is designed to be deterministic.

Given the same:

ScientificDocumentState

and:

ScientificStructureState

the builder should produce the same:

ScientificRelationshipGraph

including stable relationship IDs and deterministic ordering.

---

## Serialization

ScientificGraphNodeRef provides:

to_dict()

ScientificRelationshipCandidate provides:

to_dict()

ScientificRelationshipGraph provides:

to_dict()
to_json()

JSON serialization is deterministic.

The schema version is:

scientific-relationship-graph-v0.1

Enum values are serialized as plain values rather than Python enum
representations.

---

## Validation

The graph contracts reject:

- empty node IDs
- empty document IDs
- empty relationship IDs
- source document mismatch
- target document mismatch
- graph document mismatch
- self relationships
- missing provenance
- blank provenance
- duplicate provenance IDs
- duplicate relationship IDs
- confidence outside [0, 1]
- incompatible document and structure states

---

## Explicit Non-Goals

Scientific Relationship Graph v0.1 does NOT:

- verify scientific relationships
- establish hypothesis support
- establish hypothesis contradiction
- establish causal relationships
- reproduce experimental results
- evaluate statistical significance
- interpret p-values
- verify datasets
- verify citations
- interpret figures
- interpret plots
- interpret tables
- interpret equations
- infer cross-page semantic relationships
- resolve coreference
- resolve scientific entities
- perform external literature search
- create CommonEvidenceState
- update belief state
- modify Reasoner Orchestration
- modify Dynamic Reasoning Connectome
- modify live_app.py
- use an LLM
- download a neural model

---

## Multimodal Relationships

The graph vocabulary already permits future relations such as:

Caption
→ DESCRIBES
→ Figure

Equation
→ MODELS
→ Variable

Result
→ SUPPORTED_BY
→ Figure

Result
→ SUPPORTED_BY
→ Table

However, v0.1 does not infer these relationships yet.

Scientific Document Input Core currently provides the structural
substrate, but multimodal semantic linking requires additional
evidence and must not be fabricated from document layout alone.

These relationships may be added in later graph versions or
Mini-Lab stages.

---

## CONTRADICTS Semantics

CONTRADICTS exists in the relationship vocabulary.

ScientificRelationshipBuilder v0.1 does not automatically generate
CONTRADICTS edges.

Determining contradiction requires stronger semantic evidence than
same-page structural proximity.

Therefore contradiction inference is intentionally deferred.

---

## Relationship to Scientific Structure Model

Scientific Structure Model answers:

"What scientific structures might be present?"

Scientific Relationship Graph answers:

"Which candidate structures might be related, and how?"

Neither answers:

"Is this scientifically true?"

That distinction is fundamental to Mini-Lab.

---

## Relationship to Common Evidence State

Scientific Relationship Graph does not directly mutate or create:

CommonEvidenceState

That integration is reserved for:

14P.4 — CommonEvidenceState Bridge

The future bridge must preserve distinctions among:

- observed document content
- reported result
- author interpretation
- AISTHESIS inference
- established fact
- unknown

A relationship graph edge must never silently become verified
evidence during that conversion.

---

## Relationship to Reasoning Connectome

Scientific Relationship Graph represents candidate relationships
inside a scientific document.

Dynamic Reasoning Connectome routes reasoning components.

These are different graph concepts.

Canonical distinction:

Scientific Relationship Graph
= relationships represented about scientific content

Dynamic Reasoning Connectome
= routing among reasoning processes

The Scientific Relationship Graph must not be described as a
biological connectome.

---

## Testing

Scientific Relationship Graph v0.1 has dedicated tests covering:

- graph node contracts
- immutability
- node identity validation
- relationship contracts
- relationship direction
- document consistency
- self-edge rejection
- provenance requirements
- confidence bounds
- relationship vocabulary
- relationship statuses
- graph contracts
- duplicate relationship IDs
- deterministic serialization
- schema serialization
- document mismatch rejection
- hypothesis → tested_by → method
- method → uses → dataset
- method → measures → variable
- result → about → variable
- result → supports → hypothesis
- claim → supported_by → result
- heuristic status preservation
- absence of fabricated confidence
- same-page constraint
- unknown-page handling
- unlisted relationship suppression
- provenance merging
- deterministic relationship identity
- inference-method preservation
- empty graph behavior
- document identity preservation

Dedicated implementation tests:

60 passed

Passing tests demonstrate engineering correctness against the
implemented contracts.

They do not demonstrate scientific relationship-extraction
accuracy on real scientific literature.

---

## Validation Status

🧩 IMPLEMENTED

The graph contracts and conservative builder exist.

🧪 TESTED

60 dedicated tests pass.

📊 NOT YET TASK-LEVEL EVALUATED

The graph has not yet been benchmarked against manually annotated
scientific relationship ground truth.

🏆 NOT YET EXPERIMENTALLY SUPPORTED

No experiment currently demonstrates that the graph improves
downstream scientific reasoning.

---

## Canonical Statements

> A relationship candidate is not a verified scientific relationship.

> A graph edge is not a scientific fact.

> Same-page proximity is not semantic proof.

> A SUPPORTS edge does not establish scientific support.

> Repeated edges do not automatically strengthen evidence.

> Inference confidence is not scientific truth probability.

> Scientific Relationship Graph represents scientific content;
> Dynamic Reasoning Connectome routes reasoning.

> Scientific Relationship Graph v0.1 provides a conservative,
> provenance-preserving relational substrate for later Mini-Lab
> evidence integration.

---

## Current Mini-Lab Pipeline

Scientific PDF
        ↓
Scientific Document Input Core v0.1
        ↓
ScientificDocumentState
        ↓
Scientific Structure Model v0.1
        ↓
ScientificStructureState
        ↓
Scientific Relationship Graph v0.1
        ↓
ScientificRelationshipGraph
        ↓
[future]
CommonEvidenceState Bridge

---

## Next Milestone

14P.4 — CommonEvidenceState Bridge

The next milestone will determine how scientific document elements,
structure candidates, and relationship candidates can enter the
existing AISTHESIS evidence architecture without collapsing:

observation

reported result

author interpretation

AISTHESIS inference

and scientific fact

into the same epistemic category.