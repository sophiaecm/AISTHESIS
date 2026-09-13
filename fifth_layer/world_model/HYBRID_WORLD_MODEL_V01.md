# Hybrid learned + explicit physical world model v0.1

Status: **ENGINEERING MILESTONE**. This is **not SCIENTIFICALLY VALIDATED**.

| Category | Milestone status |
| --- | --- |
| IMPLEMENTED | Immutable contracts and deterministic integration |
| TESTED | Synthetic engineering correctness and regression tests |
| EXPERIMENTALLY EVALUATED | NOT YET TASK-LEVEL EVALUATED |
| SUPPORTED | NOT YET SUPPORTED by experimental evidence |

Learned latent representation is evidence-like internal signal,
not an observation and not a physical fact.

HybridWorldState does not claim that learned representations
improve prediction.

## Opt-in API

```python
from fifth_layer.world_model.hybrid_world_state import LearnedRepresentationSignal
from fifth_layer.world_model.hybrid_world_state_builder import HybridWorldStateBuilder

# physical is an existing LatentPhysicalState; constraints is an existing
# PhysicsConstraintBundle or PhysicsTransitionAssessment (or None).
signal = LearnedRepresentationSignal(
    source_model='external-model', timestamp=physical.timestamp,
    session_id=physical.session_id, scene_id=physical.scene_id,
    representation_id='external-representation-1', feature_dimension=1024,
    provenance={'producer': 'external-adapter'},
    representation_reference='opaque://external-representation-1',
)
hybrid = HybridWorldStateBuilder().build(physical, constraints, signal)
without_learned_input = HybridWorldStateBuilder().build(physical, constraints)
serialized = hybrid.to_json()
```

The explicit input is LatentPhysicalState v0.1 because it already carries
session and coordinate-frame identity. A caller with PhysicalWorldState first
uses the existing LatentPhysicalStateBuilder. This integration does not rebuild
or infer physical attributes. The frozen source objects are retained unchanged.
Constraints may include a complete retrospective transition assessment; its
history and provenance remain inspectable. The caller remains responsible for
constructing valid upstream states and supplying the intended constraint set.
This container does not recompute assessments or reconcile them with the
latent state's previously summarized active constraints.

There are three separately named source families: explicit_physical,
physics_constraint, and learned_representation. Source references, provenance,
availability, and original source uncertainty are inspectable. Missing sources
are None/unavailable; an empty supplied constraint bundle is available with no
results. Missing numerical summaries remain None, distinct from zero.

## Alignment and epistemic boundaries

Learned inputs must have the exact same session, scene, and known timestamp as
the physical snapshot. Scene identity scopes the input to the intended snapshot
within a session. Older, future, and unorderable learned inputs are rejected;
v0.1 provides no tolerance or asynchronous carry-forward policy. For video clips,
timestamp means the clip's final input time in the physical snapshot's clock.
External adapters must certify that all input frames end by that time. The core
cannot verify opaque references or undisclosed upstream model context.

Constraint bundle scene/time must match. Assessment and result provenance must
declare matching session/frame identity. Nested declared source timestamps must
not exceed the snapshot, and declared session/frame mismatches are rejected.
Missing learned coordinate-frame identity is recorded as uncertainty; an
explicit mismatched frame is rejected. No coordinate transform is attempted.
An undated physical snapshot can still be wrapped without learned input.
Direct HybridWorldState construction applies the same checks as its builder.

Existing string status conventions are retained without modifying EvidenceSource
or treating the signal as sensory EvidenceItem. The learned status is fixed to
learned_signal, its source type to learned_video_representation. Neither can be
selected as an observed/confirmed status through the constructor.

Latent difference is not object motion or a hidden actor. Latent similarity is
not world-state equality. collision_possible stays a possibility, never a
confirmed collision. Unknown physical values remain unknown. Optional
pooled_norm, temporal_change_score, and reference_similarity are finite opaque
numeric summaries without physical interpretation or cross-model comparability.
No confidence fusion, scoring, hypotheses, Bayesian updates, neural adapters,
or learned-to-semantic mappings are implemented.

## Memory, serialization, and dependencies

Raw vectors, tensors, arrays, bytes, and nested learned provenance are rejected.
Only a positive feature dimension, bounded reference strings (1024 characters),
three optional scalars, and up to 32 flat provenance fields are accepted. Keys
are limited to 128 characters and metadata strings to 1024. Memory does not grow
with the represented embedding dimension; references are never dereferenced.
The existing freeze/plain conventions provide detached metadata and deterministic
JSON. Serialization necessarily includes the supplied explicit state and
constraints, so its size still scales with those existing inputs.

No V-JEPA code, model import, checkpoint, download, inference, dependency, or
production pipeline integration is included. Direct module imports follow the
existing physical/latent APIs; package-level exports are unchanged.

## Correctness checks

Run in the repository's existing pytest environment, in this order:

```powershell
.\.venv\Scripts\python.exe -m pytest evaluation/tests/test_hybrid_world_state.py -q
.\.venv\Scripts\python.exe -m pytest evaluation/tests/test_physical_world_model.py evaluation/tests/test_physics_constraints.py evaluation/tests/test_latent_physical_state.py test_world_model.py test_world_model_v02.py -q
.\.venv\Scripts\python.exe -m pytest -q
```

The hybrid tests cover availability, epistemic preservation, temporal/session/
scene/frame rejection, provenance, deterministic serialization, bounded opaque
references, invalid metadata, immutability, and existing engine assessments.
These tests establish software contract behavior, not experimental usefulness.

Engineering run: 39 hybrid tests passed; the related world-model selection passed
240 tests and 108 subtests. The broader unit-test selection passed 597 tests and
130 subtests (558 existing tests plus 39 new tests; the existing untracked
test_prediction_experience.py was included without modification).

The unrestricted pytest collection/run was interrupted without a result. Root
script-style test files include import-time external perception/model execution,
so unrestricted regression is not certified here. To reproduce the safe unit
selection without editing existing tests or pytest configuration:

```python
import ast
from pathlib import Path
import pytest

files = []
for path in sorted(Path('.').glob('test_*.py')):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    if any(
        isinstance(node, ast.ClassDef) and any(
            isinstance(method, ast.FunctionDef) and method.name.startswith('test_')
            for method in node.body
        ) or isinstance(node, ast.FunctionDef) and node.name.startswith('test_')
        for node in tree.body
    ):
        files.append(str(path))
raise SystemExit(pytest.main(['-q', 'evaluation/tests', *files]))
```

This excludes 13 script-style files: test_active_perception.py,
test_active_perception_loop.py, test_auditory.py, test_auditory_negative.py,
test_locate_adapter.py, test_occlusion_reasoner.py, test_orchestrator.py,
test_perception_fusion.py, test_semantic_conflict.py, test_smolvlm_core.py,
test_temporal.py, test_yolo_detector.py, and test_yolo_occlusion.py.
