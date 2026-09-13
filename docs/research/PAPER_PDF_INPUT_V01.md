# AISTHESIS Paper / PDF Input v0.1

## Status

🧩 IMPLEMENTED  
🧪 TESTED  
✅ REAL PDF SMOKE TEST PASSED  
📊 NOT YET REAL-PAPER / GROUND-TRUTH EVALUATED  
🏆 NOT YET EXPERIMENTALLY SUPPORTED

---

## Purpose

Paper / PDF Input v0.1 adds scientific-document input as a fourth
application input modality inside the existing AISTHESIS application.

The input surface is now:

Live Camera

Photo

Video

Paper / PDF

This does not create a separate "AISTHESIS Science" application.

Paper / PDF is another input modality of the same AISTHESIS system.

---

## Application Architecture

The application-level scientific input path is:

Paper / PDF
↓
ScientificInputAdapter
↓
ScientificDocumentParser
↓
ScientificDocumentState
↓
ScientificStructureBuilder
↓
ScientificStructureState
↓
ScientificRelationshipBuilder
↓
ScientificRelationshipGraph
↓
ScientificEvidenceBridge
↓
EvidenceBundle
↓
CommonEvidenceStateBuilder
↓
CommonEvidenceState
↓
ScientificInputSummary
↓
Mini-Lab Results Window

The application adapter coordinates existing frozen Mini-Lab
components.

It does not reimplement them.

---

## New Application Adapter

File:

fifth_layer/scientific_input_adapter.py

ScientificInputAdapter provides the application-layer entry point for
scientific PDF input.

Its responsibility is orchestration only.

It:

- validates PDF input path
- invokes ScientificDocumentParser
- invokes ScientificStructureBuilder
- invokes ScientificRelationshipBuilder
- invokes ScientificEvidenceBridge
- invokes CommonEvidenceStateBuilder
- creates a UI-safe ScientificInputSummary
- preserves document, scene, session, and evidence identity

It does not introduce new scientific interpretation rules.

---

## ScientificInputResult

The adapter returns ScientificInputResult containing:

- source_path
- ScientificDocumentState
- ScientificStructureState
- ScientificRelationshipGraph
- ScientificEvidenceBridgeResult
- CommonEvidenceState
- ScientificInputSummary

The result preserves the complete Mini-Lab engineering chain.

---

## ScientificInputSummary

The UI-safe summary exposes:

- research questions
- hypotheses
- methods
- variables
- datasets
- results
- claims
- limitations
- relationships

Summary content is derived only from existing Mini-Lab outputs.

The application layer does not independently classify or infer
scientific content.

---

## live_app.py Integration

The existing application previously exposed:

- Live Camera
- Open Photo
- Open Video

v0.1 adds:

- Open Paper / PDF

Existing Camera / Photo / Video behavior remains unchanged.

The Paper / PDF input does not enter:

- YOLO
- SmolVLM scene perception
- tracking
- temporal motion reasoning
- visual prediction evaluation
- visual confidence calibration

Scientific-document evidence follows the separate documentary evidence
path already defined by Mini-Lab.

---

## Mini-Lab Results Window

Successful PDF analysis opens a separate Tkinter Toplevel window inside
the existing AISTHESIS application.

The window displays:

Research Questions

Hypotheses

Methods

Variables

Datasets

Reported Results

Author Claims

Limitations

AISTHESIS-Inferred Relationships

The window is scrollable.

Empty sections remain visible as:

Not detected

Missing information is therefore not silently hidden.

---

## Epistemic UI Boundary

The Mini-Lab window explicitly warns:

Mini-Lab extracts candidate scientific structure and relationships.

Displayed content is not independently verified scientific fact.

The UI distinguishes:

Reported Result

Author Claim

AISTHESIS-Inferred Relationship

The UI does not label extracted content as:

- verified
- proven
- confirmed
- established scientific truth

---

## Core Epistemic Boundaries

The following distinctions remain mandatory:

Document extraction != scientific understanding

Structure candidate != verified scientific structure

Relationship candidate != verified scientific relationship

Reported result != reproduced result

Author claim != established fact

AISTHESIS inference != author statement

Evidence aggregation != belief formation

End-to-end execution != scientific validation

---

## Relationship Safety Boundary

A relationship such as:

Result
→ SUPPORTS
→ Hypothesis

remains:

aisthesis_inference

The application UI does not convert graph vocabulary into scientific
proof.

Likewise:

CONTRADICTS

does not become verified contradiction.

---

## Confidence Boundary

Paper / PDF Input v0.1 does not create truth confidence.

Documentary evidence preserves the existing Mini-Lab confidence
semantics.

No UI action upgrades candidate confidence into scientific truth
probability.

---

## Error Handling

Rejected PDF input does not crash the main AISTHESIS application.

Application-level errors are displayed through a Tkinter error dialog.

The application does not fabricate fallback scientific results.

A failed PDF analysis remains a failed analysis.

---

## PDF Support

The existing ScientificDocumentParser remains responsible for PDF
ingestion.

It provides existing protections for:

- unsupported extensions
- missing files
- unreadable files
- encrypted PDFs
- corrupted PDFs
- page-count limits
- file-size limits
- text limits
- parser failures

Paper / PDF Input v0.1 does not bypass these protections.

---

## Testing

Dedicated adapter tests cover:

- PDF-path validation
- ScientificInputResult contracts
- document identity preservation
- scene identity
- session identity
- timestamp preservation
- structure-state production
- relationship-graph production
- ScientificEvidenceBridge integration
- CommonEvidenceState integration
- documentary evidence family
- UI-safe summary creation
- structure summary fields
- relationship summary fields
- reported-result preservation
- author-claim preservation
- hypothesis non-promotion
- AISTHESIS-inference preservation
- support-semantics safety
- contradiction-semantics safety
- absence of fabricated confidence
- deterministic output

Mini-Lab end-to-end tests also pass.

The broader Mini-Lab regression suite passes.

live_app.py passes Python compilation.

---

## Real PDF Smoke Test

Paper / PDF Input v0.1 was exercised with a real PDF through the actual
application UI.

Observed application behavior:

- AISTHESIS remained running
- Paper / PDF selection succeeded
- ScientificInputAdapter executed
- a document identity was created
- Mini-Lab results window opened
- extracted candidate sections were displayed
- missing sections displayed "Not detected"
- the scrollable results interface worked

This demonstrates application integration against a real PDF.

It does not establish scientific extraction accuracy.

---

## Real-Paper Observation

The real PDF smoke test produced sparse scientific-structure
extraction.

Some candidate categories were detected while others were not.

This is consistent with the current conservative lexical
Scientific Structure Model v0.1.

This observation is not treated as a solved problem.

It indicates that future real-paper ground-truth evaluation is
necessary.

---

## Known v0.1 Limitation — Sparse Real-Paper Extraction

Scientific Structure Model v0.1 is lexical and conservative.

Real scientific papers may express:

- hypotheses
- methods
- variables
- results
- claims
- limitations

without using the exact lexical patterns currently recognized.

Therefore a successfully parsed scientific paper may produce only a
small number of structure candidates.

Paper / PDF Input v0.1 does not attempt to solve that limitation at the
application layer.

Scientific extraction quality must be improved and evaluated in a
future scientific-structure milestone.

---

## Known v0.1 Limitation — Relationship Sparsity

Scientific relationships depend on available structure candidates.

If upstream structure detection is sparse, the relationship graph may
contain few or zero edges.

Zero detected relationships therefore does not mean that the paper has
no scientific relationships.

It means that v0.1 did not construct relationship candidates under its
current rules.

---

## UI Encoding

During the first Windows smoke test, several Unicode punctuation
characters were rendered incorrectly.

The UI was changed to use ASCII-safe punctuation in Mini-Lab labels.

This was an application-layer rendering issue.

It was not a scientific-document parsing issue.

---

## Frozen Core

Paper / PDF Input v0.1 consumes, but does not rewrite:

14P.1 Scientific Document Input Core v0.1

14P.2 Scientific Structure Model v0.1

14P.3 Scientific Relationship Graph v0.1

14P.4 Scientific Evidence Bridge v0.1

14P.5 Mini-Lab End-to-End v0.1

The frozen Mini-Lab core remains unchanged.

---

## Explicit Non-Goals

Paper / PDF Input v0.1 does NOT:

- verify scientific claims
- reproduce experiments
- verify statistical analyses
- establish hypothesis support
- establish contradiction
- calculate consensus
- search external literature
- compare multiple papers
- perform citation verification
- solve structure-extraction accuracy
- solve relationship-extraction accuracy
- interpret every scientific figure
- interpret every table
- interpret every equation
- introduce an LLM
- download a new neural model
- modify Camera / Photo / Video reasoning
- modify the frozen Mini-Lab core

---

## Validation Status

### 🧩 IMPLEMENTED

Paper / PDF is available as a fourth AISTHESIS application input.

### 🧪 TESTED

Adapter tests, end-to-end tests, broader Mini-Lab regression, and
live_app.py compilation pass.

### ✅ REAL PDF SMOKE TEST PASSED

A real PDF was successfully processed through the application and
displayed in the Mini-Lab results window.

### 📊 NOT YET REAL-PAPER / GROUND-TRUTH EVALUATED

Scientific structure and relationship accuracy have not yet been
measured against annotated real-paper ground truth.

### 🏆 NOT YET EXPERIMENTALLY SUPPORTED

No experiment currently demonstrates that Mini-Lab improves scientific
reasoning performance relative to appropriate baselines.

---

## Canonical Statements

> Paper / PDF is an input modality, not a separate AISTHESIS product.

> Application integration is not scientific validation.

> A parsed paper is not an understood paper.

> A detected structure is not a verified scientific structure.

> A detected relationship is not a verified scientific relationship.

> A reported result is not a reproduced result.

> An author claim is not an established fact.

> Missing extraction remains missing.

> Passing a real PDF smoke test demonstrates application integration,
> not scientific understanding accuracy.

---

## Current Input Surface

AISTHESIS

[ Live Camera ]

[ Open Photo ]

[ Open Video ]

[ Open Paper / PDF ]

---

## Next Step

14P Mini-Lab application integration is complete for v0.1.

The next scientific work should not be hidden inside the application
layer.

Future work should separately evaluate and improve real-paper
scientific understanding using annotated datasets and controlled
benchmarks.