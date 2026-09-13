# Scientific Document Input Core v0.1

## Status

ENGINEERING MILESTONE

- 🧩 IMPLEMENTED
- 🧪 TESTED
- 📊 NOT YET TASK-LEVEL EVALUATED
- 🏆 NOT YET EXPERIMENTALLY SUPPORTED

## Purpose

This milestone provides the structural input substrate for AISTHESIS Mini-Lab.

The goal is to ingest scientific PDF documents and preserve their structural contents in a deterministic, inspectable representation.

Document extraction is not scientific understanding.

Detected structure is not scientific meaning.

Extracted text is not verified scientific fact.

## Pipeline

PDF / Scientific Paper
        ↓
Document loading
        ↓
Page extraction
        ↓
Structural document elements
        ├── text
        ├── image region
        ├── equation-ready
        ├── table-ready
        ├── caption-ready
        └── unknown
        ↓
ScientificDocumentState

## ScientificDocumentState

ScientificDocumentState is intended to preserve:

- stable document identity
- source metadata
- document hash
- page count
- page boundaries
- page-level elements
- deterministic element identifiers
- extracted text
- spatial information where available
- extraction warnings
- parser metadata

The state must remain inspectable and serializable without embedding raw PDF bytes, parser objects, model objects, file handles, or large tensors.

## Structural Elements

The document model is designed to represent multimodal scientific-document structure.

Possible element categories include:

- text
- title
- section heading
- image region
- equation
- table
- caption
- reference
- footnote
- unknown

Support for an element type does not imply scientific interpretation.

For example:

- an equation-like element is not an interpreted equation
- an image region is not automatically a scientific figure
- a table-like region is not automatically understood data
- extracted caption text is not automatically linked to a figure

## Text Extraction

Text extraction preserves document and page boundaries.

Text is treated as document content, not as verified scientific knowledge.

Extracted author statements are not automatically treated as established facts.

## Image Regions

Embedded image regions may be represented structurally when exposed by the PDF parser.

An image region does not automatically mean:

- scientific figure
- plot
- chart
- experimental evidence

Those classifications belong to later Mini-Lab stages.

## Equation Support

The document-state architecture can represent equation-like elements.

Equation representation in v0.1 is structural only.

No mathematical interpretation, variable grounding, or scientific inference is performed.

## Table Support

The architecture can represent table-like structures.

No claim is made that table semantics, variable relations, experimental groups, or statistical meaning are understood in this milestone.

## Caption Detection

Caption detection may use an explicit opt-in heuristic such as prefixes including:

- Figure
- Fig.
- Table

Heuristic caption detection must remain explicitly marked as heuristic.

Caption detection does not create a figure-caption relationship automatically.

## Determinism

The same document parsed with the same configuration should produce deterministic structural output.

Stable identifiers should not depend on randomness.

## Immutability

Final document-state objects should be protected from accidental mutation where practical.

Caller-side mutation should not silently alter previously produced ScientificDocumentState objects.

## Serialization

ScientificDocumentState is intended to support deterministic serialization.

Serialized state must not contain:

- raw PDF binary payloads
- live file handles
- parser instances
- model objects
- GPU objects
- raw tensors

## Error Handling

The document input layer should explicitly handle conditions such as:

- missing file
- unsupported extension
- malformed PDF
- corrupted PDF
- encrypted PDF
- unreadable pages
- partial extraction failures
- resource-limit violations

Failures must not be silently converted into successful empty analyses.

## Resource Limits

The parser may enforce bounded resource usage such as:

- maximum file size
- maximum page count
- bounded extracted text
- bounded metadata

These are engineering safeguards and not scientific-analysis rules.

## Dependency

The v0.1 scientific-document parser uses PyMuPDF as an isolated document-ingestion dependency.

PyMuPDF provides PDF parsing, page geometry, text-block extraction, and embedded image-region access without requiring an OCR model or external multimodal model.

PyMuPDF is an ingestion substrate only.

It is not the final scientific document understanding system.

## Epistemic Boundaries

The following distinctions must be preserved:

document extraction ≠ scientific understanding

detected structure ≠ scientific meaning

extracted text ≠ verified scientific fact

detected image region ≠ scientific figure interpretation

equation representation ≠ equation understanding

table representation ≠ table interpretation

caption detection ≠ figure-caption relationship

author statement ≠ established fact

## What v0.1 Does Not Do

This milestone does not implement:

- research-question extraction
- hypothesis extraction
- method interpretation
- variable extraction
- dataset reasoning
- result interpretation
- claim evaluation
- limitation analysis
- scientific relationship graphs
- contradiction detection
- evidence-support graphs
- figure interpretation
- equation interpretation
- table interpretation
- LLM summarization
- OCR model inference
- external scientific model downloads

## Mini-Lab Roadmap

14P.1 Scientific Document Input Core
        ↓
14P.2 Scientific Structure Model
        ↓
14P.3 Scientific Relationship Graph
        ↓
14P.4 CommonEvidenceState Bridge
        ↓
14P.5 End-to-End Mini-Lab Evaluation
        ↓
14P.6 live_app.py Paper/File Input

## Future Compatibility

### 14P.2 Scientific Structure Model

The next stage will use ScientificDocumentState to identify structures such as:

- research question
- hypothesis
- methods
- variables
- datasets
- results
- claims
- limitations

### 14P.3 Scientific Relationship Graph

Later stages may represent relations such as:

hypothesis → tested_by → experiment

experiment → uses → dataset

result → supported_by → figure/table

caption → describes → figure

equation → models → relation

Stable document element identities established in v0.1 are intended to support these later relationships.

### live_app.py

The future AISTHESIS interface is intended to support:

- Live Camera
- Photo
- Video
- Paper / PDF

This milestone does not modify live_app.py.

UI integration belongs to a later Mini-Lab milestone.

## Scientific Status

This milestone establishes document-ingestion infrastructure only.

It does not demonstrate that AISTHESIS understands scientific papers, improves scientific reasoning, or produces experimentally validated scientific conclusions.
