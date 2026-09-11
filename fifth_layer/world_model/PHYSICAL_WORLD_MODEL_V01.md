# Physical World Model v0.1

Opt-in, deterministic image-plane physical state built from `SceneState` and
`EvidenceBundle`. No production pipeline is connected or modified. Experience
Learning v0.1, sensory evaluation, prediction evaluation and reasoners are unchanged.
No training, embeddings, neural models, downloads or additional dependencies.

## API

```python
from fifth_layer.world_model.physical_state_builder import PhysicalStateBuilder

state = PhysicalStateBuilder().build(scene, evidence, previous_scene=previous)
velocity = state.objects[0].attributes['velocity']
print(velocity.value, velocity.status, velocity.derived_from, velocity.rule)
serialized = state.to_json()
```

`evidence` and `previous_scene` are optional. Omitted evidence is projected using
existing evidence providers; no reasoner is called. Models are frozen dataclasses
with detached immutable nested mappings. `to_dict()` produces detached JSON data;
`to_json()` uses canonical sorted keys, finite numbers and fixed separators.

* `PhysicalAttribute`: value, status, source references, derivation rule and units.
  Status is `observed`, `estimated`, `possible`, `unknown`, or `unavailable`.
  Observed detector labels/confidence describe source reports, not ground truth.
* `PhysicalObjectState`: deterministic identity, snapshot timestamp, attributed
  physical fields and original structured observation.
* `PhysicalRelation`: observed endpoint identities, relation type and attributed
  derivation. Endpoints must exist in the current snapshot.
* `PhysicalWorldState`: scene/time, sorted objects/relations, evidence references,
  uncertainty metadata, builder policy and provenance, schema `physical-world-0.1`.

Every non-null attribute requires provenance. Derived values additionally require
a rule. Evidence used for attributes is retained verbatim in world provenance,
including raw motion values and source confidence. Other evidence IDs are listed
as context only. No confidence aggregation or invented probability occurs.

## Geometry, identity and motion

Coordinates are **pixels**, x increasing right, y increasing down. `box_xyxy` and
`bbox` mean xyxy; legacy `box` means xywh. Center and size derive from the box,
or an explicitly supplied center is retained when the box is absent. This is
projected geometry, never depth, 3D shape or metric world coordinates.

Only `observed_objects` create physical objects. Track identity preserves type
(integer 1 differs from string "1"). Duplicate tracks are rejected. Without a
track, identity is scoped to scene and observation index; there is no automatic
cross-frame association by label or nearest neighbor. Track IDs are session-local;
callers must keep tracking sessions separate and never compare IDs across sessions.

Matching track observations in an explicitly supplied earlier scene support
displacement. The caller guarantees the same tracking session and coordinate
frame; image dimensions must agree. A class change blocks history association.
Velocity is a finite-difference image-plane estimate only when timestamps give a
positive interval. Explicit positive `dt`/`delta_time` in a current motion record
also supports displacement/interval velocity. Stored velocity alone is insufficient;
it remains in raw evidence and cannot establish timing by itself.

Existing `motion_evidence` classification is retained (including its source
deadband); temporal forecasts are excluded. For history-only classification,
`minimum_motion_pixels` defaults to `None`: classification stays unknown. Callers
may configure a nonnegative pixel threshold; displacement magnitude <= threshold
is stationary. The threshold is recorded in provenance. This avoids silently
inventing a deadband. Raw displacement is never zeroed by classification.
Direction is the source direction label when available, otherwise a normalized
nonzero displacement vector; sub-deadband jitter can have a direction.

Current motion/occlusion evidence must match the current timestamp and object
association. Ambiguous multiple motion records are not selected arbitrarily.
Untimed history requires strictly increasing `snapshot_sequence_id` values and
cannot establish velocity. Explicit object observation timestamps must equal their
scene timestamp: asynchronous sensor fusion is outside v0.1. Future history,
future evidence timestamps and future nested measurement/provenance timestamps
are rejected. Forecast payloads are ignored. Caller-supplied timestamps and source
labels are trusted contracts; the builder cannot authenticate sensor reports.

## Relations and epistemic safeguards

Center comparisons produce left/right/above/below relations. Positive bbox
intersection produces overlap, without causal interpretation. Frame truncation
uses the existing perception rule: touching or crossing any image boundary.
Truncation needs dimensions or explicit stored boundary evidence; absence stays
unknown. Stored occlusion overlap evidence supports only `possible` occlusion,
never a confirmed hidden state or new object. No overlap does not prove visibility
is complete. `visibility_state=observed` says an observation exists, not that the
entire object is visible.

Explicit relation input is an observed sensory `EvidenceItem` with type
`contact_observation` or `support_observation`, `epistemic_status='observed'`,
and value `{observed: True, subject_object_id: <index>, target_object_id: <index>}`.
Indices refer to the current `observed_objects` sequence. Both endpoints must be
observed now. Output is `contact_possible`/`support_possible`, without dynamics.
These are optional extension contracts; existing sensory evaluators are untouched.
Expected, inferred and unavailable sensations cannot establish these relations.
Object-level contact/support stay unknown; explicit pairwise evidence is in relations.
`near` is omitted because this builder defines no proximity threshold.

Missing detector observations produce no current object entry and explicitly leave
disappearance unknown. Predicted tracks do not become observations. Semantic/VLM
text, experience, generic physics reasoner output and future trajectories cannot
define physical attributes. Unknown velocity, depth, mass, forces, friction,
contact and support remain null. No hidden actors, collisions or 3D facts are
fabricated. This is not a physics engine or a new prediction/learning system.

## Offline Case A/B utility

```powershell
.\.venv\Scripts\python.exe -m evaluation.physical_world_model results_case_a_fifth_layer_sensory.json --output results_case_a_physical_world.json
.\.venv\Scripts\python.exe -m evaluation.physical_world_model results_case_b_fifth_layer_sensory.json --output results_case_b_physical_world.json
```

Output must be a new `*_physical_world.json` file. Exclusive creation prevents
overwrites. Source bytes are SHA-256 referenced. The utility reads saved
`evaluation-0.1` records, including plain fifth-layer and combined variants,
projects `fifth_layer_input` observed detections/motion/occlusion geometry, and
isolates sessions by mode and source hash. VLM outputs, ground truth, annotations,
reasoner outputs and stored hypotheses are never used. Failed/non-observation
records are diagnosed. Observation/frame timestamp mismatch is rejected. Previous
scenes are drawn only from earlier snapshots in the same session.

Architectural validation on the saved fifth-layer sensory files:

| Stored case | Snapshots | Ball observations | Ball displacement estimates | Person observations | First person observation |
| --- | ---: | ---: | ---: | ---: | --- |
| A | 20 | 9 | 8 | 14 | 3.0 seconds |
| B | 10 | 4 | 3 | 0 | none |

Ball labels are `sports ball` in the source files; first observed at 3.0 seconds
in both cases. Case A contains no person physical object before that time. Both
reports contain only geometric relation types, no collision/contact/support
relations or hidden actors. These are source-preserving architectural checks,
**not improved accuracy claims**.

## Validation

31 new unittest cases cover the requested 18 scenarios plus observed relation
contracts, immutable records, identity ambiguity, configurable deadbands,
explicit motion timing, asynchronous timestamps and nested future provenance.
All 461 automated tests across 24 modules passed under the existing `.venv`
(430 existing + 31 new). System Python initially lacked existing NumPy/Pillow;
the existing environment resolved that without installations.

The repository also contains 13 root `test_*.py` executable demos without unittest
cases, including SmolVLM/YOLO/LocateAnything inference at import time. They were not
executed, consistent with the prohibition on rerunning models. To reproduce the
complete automated suite while excluding those demos:

```python
import ast
from pathlib import Path
import unittest

files = sorted(Path('.').glob('test_*.py')) + sorted(Path('evaluation/tests').glob('test_*.py'))
modules = [str(p.with_suffix('')).replace('\\', '.').replace('/', '.') for p in files
           if any(isinstance(n, ast.ClassDef)
                  and any('TestCase' in ast.unparse(b) for b in n.bases)
                  for n in ast.parse(p.read_text(encoding='utf-8-sig')).body)]
result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromNames(modules))
raise SystemExit(not result.wasSuccessful())
```
