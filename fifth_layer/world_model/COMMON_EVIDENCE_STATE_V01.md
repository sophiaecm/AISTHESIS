# Common evidence / state integration v0.1

Status: **ENGINEERING MILESTONE**.

- IMPLEMENTED
- TESTED (synthetic software correctness)
- NOT YET TASK-LEVEL EVALUATED
- NOT YET EXPERIMENTALLY SUPPORTED

Evidence aggregation is not belief formation.

CommonEvidenceState preserves evidence identity, provenance,
availability, uncertainty, and epistemic status without deciding truth.

This milestone makes no scientific improvement or prediction-quality claim.

## Reused architecture

EvidenceItem already supplies identity, source, status, payload, provenance,
confidence, and support/opposition references. EvidenceBundle already rejects
duplicate IDs, checks scene identity, freezes metadata, and orders items. Both
are reused. No duplicate evidence record class is introduced.

EvidenceSource gains five opt-in members: visual, physical, physics_constraint,
learned_representation, and documentary. Existing enum values, constructors,
dataclass fields, serialized fields, and legacy validation remain unchanged.
New source types require compatible explicit epistemic statuses. Their payloads
use the existing generic freeze boundary rather than the legacy numeric
uncertainty validator: a physical uncertainty tuple is not a probability.

CommonEvidenceState adds integration context, source references, family
availability, source uncertainty indexed by original path, and integrity
information around an EvidenceBundle. Its evidence_items property exposes the
existing EvidenceItems. No package-level exports or production consumers change.

## API

```python
from fifth_layer.world_model.evidence import EvidenceItem
from fifth_layer.world_model.common_evidence_state_builder import CommonEvidenceStateBuilder

reported = EvidenceItem(
    evidence_id='paper:result:1', scene_id='review-session-scene',
    source_type='documentary', source_component='external-mini-lab',
    evidence_type='reported_result', value={'summary': 'Synthetic result'},
    timestamp=None, epistemic_status='reported_result',
    provenance={'session_id': 'review-session', 'document_reference': 'synthetic:paper'},
)
state = CommonEvidenceStateBuilder().build(
    scene_id='review-session-scene', session_id='review-session', timestamp=None,
    evidence=(reported,),
)
print(state.to_json())
```

The builder accepts an EvidenceItem, an EvidenceBundle, or an ordered sequence
of either through evidence. Existing sensory provider outputs work unchanged.
Optional hybrid_state expands into physical, constraint, and learned items;
alternatively supply physical_state (LatentPhysicalState), physics_constraints
(PhysicsConstraintBundle or PhysicsTransitionAssessment), and learned_signal
(LearnedRepresentationSignal) separately. Hybrid and standalone inputs cannot
be combined ambiguously. No input is mutated or dereferenced.

Existing evidence IDs, latent_state_id, constraint_id, and representation_id
are retained. Duplicate IDs are always rejected, including identical content
and collisions between source families. Producers must use globally distinct
IDs in a common inventory. Similar statements with different IDs remain
separate. Bundle provenance is indexed by a deterministic content reference.
Full supplied structured physical/constraint metadata stays inspectable inside
the adapted item's value; original provenance remains nested or unchanged.

## Families, availability, and status

The five new source families coexist with sensory and all eight legacy source
values (including physics, temporal, occlusion, semantic, motion, tracking,
experience). Legacy physics is kept distinct from structured physical state and
physics_constraint. Sources are grouped through source_references and the
availability map without relabelling legacy motion/tracking as observations.

Sensory observed, expected, inferred, and unavailable remain separate item
statuses within the sensory family. Availability means a non-unavailable signal
was supplied, not that a sensory observation exists. Expected sensory evidence
therefore cannot establish that sound was observed. Individual unavailable
records remain unavailable even when another item from that family is present.

Absent families have unavailable status. An explicitly present empty source can
be declared with present_families=('visual',), and a supplied empty constraint
bundle is automatically available. This is feed availability, not negative
evidence or confidence. An empty generic EvidenceBundle cannot identify which
families were queried; use present_families when that distinction is known.

Physical summary items use mixed status because their nested attributes retain
observed/estimated/possible/unknown distinctions. Constraint items use assessment
status; original finding, result status, uncertainty, and derived_from are
preserved. satisfied plus collision_possible never becomes collision. Learned
items remain learned_signal. None is retained, distinct from zero or false.
Legacy missing epistemic status remains None; no observed status is inferred.

Support/opposition references and source confidence are copied without merging,
voting, ranking, contradiction resolution, or deciding truth. Do not feed these
new transport sources to legacy hypothesis consumers assuming semantic support;
this opt-in inventory performs no downstream routing.

## Context and integrity

Scene/session mismatches are rejected wherever these exact context keys are
declared. Missing legacy session identity is flagged source_context_incomplete;
it is not silently replaced with the integration session. A source item must
belong to the inventory scene, including externally supplied documentary items.
Documents retain independent document IDs in provenance; scene_id identifies
the review context, not the original document's experimental scene.

Known evidence timestamps cannot exceed the inventory cutoff. Nested declared
source timestamps cannot exceed the enclosing source time. Unknown time is
flagged temporal_alignment_unknown; known inputs cannot be integrated against
an unknown cutoff. Older non-learned evidence is retained with its original time,
not promoted to a current observation. Learned inputs require exact known time
equality, consistent with Step 13. Explicit target_timestamp is a prediction
target, not source time, and may lie after the cutoff; its value remains opaque.
Adapters must use source timestamp fields only for actual source times, not
publication dates or hypothetical future consequences.

Frame mismatches are rejected for frame-bound sources; missing frame information
is flagged coordinate_frame_unspecified. Documentary sources require no spatial
frame. No coordinate transformation or automatic clock conversion is performed.
These checks validate declared metadata, not undisclosed upstream context.
Direct CommonEvidenceState construction applies the same inventory validation.

## Mini-Lab readiness

reported result != author claim != AISTHESIS inference != established fact.

Documentary EvidenceItems support reported_result, author_claim,
aisthesis_inference, unknown, limitation, method, measurement, dataset_reference,
figure_evidence, table_evidence, and equation_relation. evidence_type remains an
extensible category string; provenance may identify pages, figures, datasets,
or references. These are supplied classifications, not classifications computed
by AISTHESIS. No PDF/document parser, scientific reasoning, model, Bayesian
layer, truth fusion, Dynamic Connectome, or ASTRA-EFA update is implemented.

## Immutability, bounds, and serialization

The common boundary reconstructs detached existing EvidenceItems, freezes maps,
and emits canonical JSON with stable item/key ordering and no NaN. Metadata is
limited to 50,000 traversal nodes, depth 24, 16,384 characters per string, 256
characters per key, and 1 MiB serialized per boundary call, including the final
common state. Oversized input fails explicitly; it is not silently truncated.
Repeated source metadata counts toward this limit. Large inventories must be
partitioned by the caller.

Bytes, raw tensors/arrays, file handles, and opaque model objects are rejected.
Raw embedding/vector/model payload fields are forbidden; external references
are not opened. LearnedRepresentationSignal retains the stricter Step 13 limits.
Small numeric summaries and structured physical geometry remain permitted.

## Correctness execution

```powershell
.\.venv\Scripts\python.exe -m pytest evaluation/tests/test_common_evidence_state.py -q
.\.venv\Scripts\python.exe -m pytest evaluation/tests/test_hybrid_world_state.py evaluation/tests/test_physical_world_model.py evaluation/tests/test_physics_constraints.py evaluation/tests/test_latent_physical_state.py test_world_model.py test_world_model_v02.py test_sensory_evidence.py evaluation/tests/test_sensory.py -q
```

The new tests exercise 61 cases, including multi-source inventory, provenance,
identity, contradiction preservation, scientific categories, alignment, input
immutability, payload bounds, serialization, and hybrid/assessment adapters.
New tests: 61 passed. Related regression: 318 passed, 108 subtests passed.

Unrestricted pytest is not rerun: root smoke scripts have known import-time
model/perception execution (see HYBRID_WORLD_MODEL_V01.md). The safe broader
selection uses evaluation/tests plus root test files defining test functions or
test methods, exactly as documented in Step 13. It excludes the same 13 script
files and includes existing untracked test_prediction_experience.py read-only.
No claim of an unrestricted full-suite pass is made.

Safe broader regression: **658 passed, 130 subtests passed** (597 existing tests
plus 61 new tests). No commit, push, or generated-file staging was performed.
