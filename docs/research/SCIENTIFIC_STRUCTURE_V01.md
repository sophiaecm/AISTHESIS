# AISTHESIS Scientific Structure Model v0.1

## Status

🧩 IMPLEMENTED  
🧪 TESTED  
📊 NOT YET TASK-LEVEL EVALUATED  
🏆 NOT YET EXPERIMENTALLY SUPPORTED

---

## Purpose

Scientific Structure Model v0.1 is the second stage of the
AISTHESIS Mini-Lab scientific-document pipeline.

It converts the structural document representation produced by
Scientific Document Input Core v0.1 into conservative,
source-grounded scientific-structure candidates.

Pipeline:

Scientific PDF
→ ScientificDocumentState
→ ScientificStructureBuilder
→ ScientificStructureState

This milestone performs candidate extraction only.

It does not perform scientific verification, evidence evaluation,
relationship inference, or scientific truth assessment.

---

## Epistemic Boundary

The following distinctions are mandatory:

candidate structure != verified scientific structure

extracted hypothesis != supported hypothesis

extracted result != reproduced result

extracted claim != true claim

lexical match != scientific understanding

extractor confidence != truth probability

repeated statement != stronger evidence

document extraction != scientific validation

AISTHESIS must preserve these distinctions throughout later
Mini-Lab stages.

---

## Input

Scientific Structure Model v0.1 consumes:

ScientificDocumentState

from:

fifth_layer/scientific_document.py

The input preserves document identity, page structure,
document elements, source references, and extracted text.

Scientific Structure Model does not parse PDF files directly.

PDF ingestion remains the responsibility of Scientific Document
Input Core v0.1.

---

## Output

The primary output is:

ScientificStructureState

containing immutable:

ScientificStructureCandidate

objects.

Each candidate preserves:

- candidate ID
- scientific structure type
- epistemic status
- extracted text
- document ID
- source element IDs
- human-readable page number
- optional extractor confidence
- extraction method

Source provenance is mandatory.

A scientific structure candidate cannot exist without at least
one source document element reference.

---

## Supported Scientific Structure Types

v0.1 represents:

- RESEARCH_QUESTION
- HYPOTHESIS
- METHOD
- VARIABLE
- DATASET
- RESULT
- CLAIM
- LIMITATION
- UNKNOWN

These types describe candidate document structures.

They do not establish that the extracted content is scientifically
correct.

---

## Epistemic Status

ScientificStructureStatus supports:

- CANDIDATE
- EXPLICIT
- HEURISTIC
- UNKNOWN

The v0.1 lexical builder emits:

HEURISTIC

for automatically detected candidates.

It does not automatically promote candidates to EXPLICIT.

An explicit-looking sentence in a paper is still not automatically
a verified scientific fact.

---

## Conservative Extraction

ScientificStructureBuilder v0.1 uses deterministic lexical patterns.

Examples include expressions such as:

- "research question"
- "we hypothesize"
- "our hypothesis"
- "we recruited"
- "participants"
- "dependent variable"
- "dataset"
- "we found"
- "we observed"
- "we conclude"
- "limitations"

These patterns are extraction cues only.

They are not semantic proof that a sentence has been correctly
understood.

---

## Allowed Source Elements

v0.1 performs lexical candidate extraction only from selected
textual document elements:

- text_block
- title
- section_heading

It intentionally does not infer scientific claims directly from:

- caption
- image_region
- figure
- graph
- table
- equation
- reference

Those multimodal relationships belong to later Mini-Lab stages.

---

## Provenance

Every candidate retains:

document_id

and:

source_element_ids

This allows later stages to trace a scientific candidate back to
the original structural document elements.

When duplicate candidates are merged, source provenance is merged
rather than discarded.

---

## Page Semantics

Scientific Document Input Core uses zero-based page indexes
internally.

Scientific Structure Model exposes human-readable page numbers:

page_number = page_index + 1

No page identity is inferred when it is unavailable.

---

## Duplicate Suppression

v0.1 conservatively suppresses exact same-page duplicates.

Candidates are considered duplicate only when they have:

- the same scientific structure type
- the same normalized text
- the same human-readable page number

When duplicates originate from multiple source elements,
source_element_ids are merged deterministically.

Repeated text on different pages is not automatically collapsed.

This is intentional.

A repeated statement in different document locations may have
different scientific context.

Duplicate suppression does not increase confidence and does not
strengthen evidence.

---

## Determinism

Given the same ScientificDocumentState, the builder is designed to
produce the same ScientificStructureState.

Stable candidate IDs are generated from deterministic source
information.

Duplicate source IDs are sorted before merged candidate identity
is generated.

ScientificStructureState supports deterministic JSON serialization.

Serialization uses plain enum values rather than Python enum
representations.

---

## Confidence

The data model supports optional extractor confidence for future
compatibility.

The v0.1 builder assigns:

confidence = None

This is intentional.

No numerical truth probability is fabricated from lexical rules.

Future extraction models may expose model confidence, but such
confidence must remain distinct from:

- scientific truth
- evidence strength
- hypothesis support
- calibrated belief

---

## Immutability

ScientificStructureCandidate and ScientificStructureState are
immutable dataclasses.

The model rejects:

- empty candidate IDs
- empty document IDs
- empty candidate text
- missing source provenance
- blank source references
- duplicate source references
- invalid page numbers
- confidence outside [0, 1]
- duplicate candidate IDs
- candidates belonging to a different document

---

## Serialization

ScientificStructureCandidate provides:

to_dict()

ScientificStructureState provides:

to_dict()
to_json()

JSON serialization is deterministic.

The schema version is:

scientific-structure-v0.1

---

## Explicit Non-Goals

Scientific Structure Model v0.1 does NOT:

- verify research questions
- determine whether hypotheses are supported
- reproduce results
- determine whether claims are true
- infer causal relationships
- infer hypothesis-to-experiment relationships
- infer result-to-figure relationships
- interpret figures
- interpret graphs
- interpret tables
- interpret equations
- perform citation verification
- perform external literature search
- construct a scientific relationship graph
- create CommonEvidenceState
- modify Reasoner Orchestration
- modify Dynamic Reasoning Connectome
- modify live_app.py
- use an LLM
- download a neural model

---

## Relationship to Mini-Lab

Current Mini-Lab pipeline:

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
[future]
Scientific Relationship Graph

Scientific Relationship Graph is a separate milestone.

Examples of future candidate relationships include:

hypothesis → tested_by → experiment

experiment → uses → dataset

variable → measured_by → measurement

result → supported_by → figure/table

figure → derived_from → dataset

caption → describes → figure

equation → models → variable/relation

result → supports_or_contradicts → hypothesis

claim → supported_by → evidence

None of these relationships are inferred by v0.1.

---

## Relationship to Common Evidence State

Scientific Structure Model does not directly mutate or create
CommonEvidenceState.

A future Mini-Lab bridge may convert documentary scientific
structures and relationships into explicitly typed evidence.

That bridge must preserve distinctions between:

- observed document content
- reported result
- author interpretation
- AISTHESIS inference
- established fact
- unknown

---

## Testing

Scientific Structure Model v0.1 has dedicated tests covering:

- immutable contracts
- validation
- all supported structure types
- all supported statuses
- lexical candidate extraction
- research-question extraction
- hypothesis extraction
- method extraction
- variable extraction
- dataset extraction
- result extraction
- claim extraction
- limitation extraction
- unmatched text
- caption exclusion
- image-region exclusion
- source provenance
- deterministic IDs
- multiple candidates
- heuristic-status preservation
- absence of fabricated confidence
- page-number conversion
- document identity
- deterministic serialization
- schema serialization
- enum serialization
- same-page duplicate suppression
- provenance merge
- cross-page duplicate preservation
- deterministic duplicate merging

Passing unit tests demonstrate engineering correctness against the
implemented contracts.

They do not demonstrate scientific extraction accuracy on real
scientific literature.

---

## Validation Status

🧩 IMPLEMENTED

The software architecture exists.

🧪 TESTED

Unit and contract tests pass.

📊 NOT YET TASK-LEVEL EVALUATED

The extractor has not yet been benchmarked against an annotated
scientific-document dataset.

🏆 NOT YET EXPERIMENTALLY SUPPORTED

No experiment currently demonstrates that Scientific Structure
Model improves downstream scientific reasoning.

---

## Canonical Statements

> Document extraction is not scientific understanding.

> Detected structure is not scientific meaning.

> Extracted text is not verified scientific fact.

> A structure candidate is not a verified scientific structure.

> Repetition does not automatically increase evidence strength.

> Extractor confidence is not scientific truth probability.

> Scientific Structure Model v0.1 provides a conservative,
> provenance-preserving structural substrate for later Mini-Lab
> reasoning.

---

## Next Milestone

14P.3 — Scientific Relationship Graph v0.1

The next milestone will represent explicit, provenance-preserving
candidate relationships among scientific structures and multimodal
document elements without treating inferred relationships as
scientific facts.