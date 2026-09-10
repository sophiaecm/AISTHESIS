# Sensory evidence integration 0.1

Implemented locally against commit `2b0634d`, tag `pre-sensory-baseline-v1`.
The integration is opt-in. It does not claim improved prediction quality.

## Repository inspection and actual interfaces

Inspected the repository tree and `reasoners/auditory.py`, `physics.py`,
`orchestrator.py`, `sensor_fusion.py`; world-model `evidence.py`,
`evidence_providers.py`, `hypothesis_generator.py`, `adapter.py`, `scene_state.py`,
`_structured.py`, `__init__.py`; evaluation `runner.py`, `harness.py`,
`adapters.py`, `README.md`, and its harness tests; existing world-model tests
and the root test scripts for model/service side effects. Repository-wide
search found no tactile, thermal or kinesthetic implementation. Inspection
preceded edits. No repository AGENTS.md was found.

| Modality | Actual implementation and inputs | Outputs and epistemic interpretation |
| --- | --- | --- |
| Auditory | `fifth_layer.reasoners.auditory.AuditoryReasoner.infer_expected_consequences(WorldState)` reads `detections`, `motion_evidence`, `scene_description` | `ExpectedConsequences.predictions`: `auditory_state`, `auditory_consequences`, `strongest_auditory_consequence`, `auditory_uncertainty`, `auditory_modality_observed=False`, `reasoning_mode=cross_modal_expected_consequence`. Candidates: `consequence`, `probability`, `uncertainty`, `evidence`, `source_object`, `observed=False`. These are heuristic expectations, not measurements or calibrated probabilities. No confidence field is emitted. |
| Tactile / force | No reasoner exists | Explicit unavailable-observation record; no fabricated consequence adapter. |
| Thermal | No reasoner exists | Explicit unavailable-observation record; no fabricated consequence adapter. |
| Kinesthetic | No reasoner exists | Explicit unavailable-observation record; no fabricated consequence adapter. |

Auditory latent output uses `auditory_modality_observed`, `reasoning_mode`,
`auditory_latent_state`, `auditory_event_possible`, `auditory_uncertainty` and,
when candidates exist, `most_likely_auditory_consequence`, `auditory_probability`,
`auditory_evidence`, `auditory_source_object`. Future output copies these and
adds `auditory_future_event`, `auditory_event_probability` and, for a consequence,
`prediction_type=expected_sensory_consequence`. These summaries lose candidate
detail: the adapter accepts expected output or its mapping, not latent/future
substitutes. No native timestamps or track IDs are emitted by this reasoner.

Auditory depends on structured visual/motion inputs and semantic text, but calls
neither physics nor a VLM nor another reasoner. In particular, semantic impact
words can produce `contact_or_impact_sound_possible`. Conversion preserves this
as an unverified expected consequence; it never establishes physical impact.

`PhysicsReasoner` reads `position`, `velocity`, optional `dt` and `occlusion_zone`.
Its expected output contains `expected_next_position`, optionally
`enters_occlusion_zone` and `hidden_interaction_possible`. Latent output can add
`latent_physical_risk`; future output can add `predicted_event`. It supplies no
contact, force, temperature, sound, or biological-motion measurements.

`AisthesisOrchestrator.analyze(WorldState)` returns `semantic_conflict`,
`active_perception`, and `occlusion`, `scene`, `temporal`, `sensor_fusion` branches
with `expected`, `latent`, `future`. It invokes neither auditory nor physics.
`SensorFusionReasoner` is an existing hidden-actor/occlusion evidence fusion
reasoner, not an implementation of the missing sensory modalities. Its generic
occlusion and semantic evidence are not additional hidden-actor evidence.

## Exact contract and integration

The existing frozen `EvidenceItem` gains four trailing, defaulted fields:

```python
modality: str | None = None
epistemic_status: str | None = None
supporting_evidence_ids: tuple[str, ...] = ()
opposing_evidence_ids: tuple[str, ...] = ()
```

`EvidenceSource.SENSORY = 'sensory'` is added. No parallel evidence architecture
or new epistemic enum is introduced. Legacy constructors still work unchanged;
serialized legacy EvidenceItems now include these four defaults.

Sensory items require modality `auditory`, `tactile`, `thermal`, or `kinesthetic`
and status `observed`, `inferred`, `expected`, or `unavailable`. These fields are
invalid on non-sensory sources. An explicit `value.observed` flag must agree with
status. Unavailable items have exactly `value={'observation_available': False}`,
unknown confidence, and no support, contradiction or evidence-ID links. This
prohibits representing silence, no contact, or no heat as missing evidence.

Existing `value`, `confidence`, `source_component`, `timestamp`, `track_id`,
`object_id`, `provenance`, `supports` and `contradicts` retain their roles.
`supports`/`contradicts` name hypothesis types; the two new ID tuples reference
supporting/opposing evidence. Source verbal evidence labels remain in `value`.
The auditory provider retains source probability as a labelled legacy heuristic
and never copies it into confidence. Explicit optional confidence stays unchanged.

The provider preserves candidate timestamps (then envelope/caller/scene fallback),
optional explicit object/track identity and source provenance. `source_object`
is a class label and is never converted into identity. Source path, timestamp,
scene provenance, adapter version and score/physics policy are retained.
Candidate order and evidence-label order do not change semantic output or IDs;
duplicates are merged without accumulating confidence. Inputs are detached and
immutable through the existing evidence model.

```python
from fifth_layer.world_model import collect_evidence, MultiHypothesisGenerator

# auditory_output is already produced by the existing reasoner.
bundle = collect_evidence(scene, auditory=auditory_output, include_sensory=True)
hypotheses = MultiHypothesisGenerator().generate(scene, bundle)
```

`include_sensory=True` adds coverage records even when output is missing.
Supplying auditory output also opts in. Omitting both preserves the original
collection behavior. Auditory expectations coexist with unavailable auditory
observations: a possible sound does not imply that a microphone exists.

Future callers can explicitly construct observed sensory EvidenceItems with a
measurement source and source claim. There is no microphone or other physical
sensor adapter in this change. Auditory conversion rejects observed flags/statuses
instead of treating its heuristic output as measurement. The contract can express
inferred tactile/thermal/kinesthetic evidence, but no adapters fabricate it.

## Physical constraints and hypothesis behavior

Sensory items are excluded from candidate generation and association-group
creation. They can only annotate candidates already produced by existing rules.
They require an explicit matching object/track association and matching source
time. Expected/inferred support also requires an evidence ID resolving to
same-associated physics, motion or temporal evidence supporting that candidate.
Physical contradiction blocks sensory support. Inferred opposition requires an
explicit opposing physical evidence ID; explicitly linked observed opposition
can oppose an existing candidate. Dangling, stale, semantic-only and occlusion-only
anchors cannot establish inferred support.

Neither supporting expectations nor repeated correlated consequences raise base
confidence or compatibility score. Explicit opposition uses the existing single
halving rule, retaining alternatives. Sensory status, modality, IDs and provenance
are included in hypothesis score-input provenance. Ranking remains uncalibrated.
No hidden actor, collision, contact, fall, danger or new event rule was added.
Sensory absence, semantic ball mentions and generic occlusion cannot create actors.

Native auditory candidates lack explicit physical anchors and identity. They
therefore improve coverage only, leaving hypotheses unchanged. Caller-supplied
links are retained and checked, not guessed from class names or semantic labels.
No valid/linked sensory evidence gives identical HypothesisSet values to the
pre-sensory path, including empty scenes. No geometry or physical values are made up.

## Manual controlled evaluation after review

The existing evaluation runner, mode boundaries and textual comparison semantics
are unchanged. New `evaluation.sensory` postprocesses saved successful Fifth Layer
records, calls only the existing deterministic auditory reasoner on the original
per-frame input, and appends `sensory_world_model`. VLM-only/rejected records are
not enriched. Original claims, comparisons, source manifests and timings remain
intact; timings describe the original harness run, not enrichment. New data are
not silently counted as added textual claims or measured scientific improvement.

Run the SAME controlled videos manually with stride 12, the original start/stop
range, threshold, devices, model snapshots and tracking defaults. The example
below uses the actual filenames found locally and the baseline's CUDA/0.4 config.
No video or neural evaluation was run during this implementation.

```powershell
$vlmSnapshot = 'C:/Users/ecmtu/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467'
foreach ($case in @('a', 'b')) {
    $video = if ($case -eq 'a') { 'test_videos/case_a_ball_human.mp4.mp4' } else { 'test_videos/case_b_ball_no_human.mp4.mp4' }
    .venv/Scripts/python.exe -m evaluation.runner $video --kind video --start 0 --stride 12 --mode FIFTH_LAYER_ONLY --detector-model yolo11n.pt --detector-device cuda --detector-confidence 0.4 --output "results_case_${case}_fifth_layer_sensory_raw.json"
    if ($LASTEXITCODE -ne 0) { throw 'Harness run failed' }
    .venv/Scripts/python.exe -m evaluation.sensory "results_case_${case}_fifth_layer_sensory_raw.json" --output "results_case_${case}_fifth_layer_sensory.json"
    if ($LASTEXITCODE -ne 0) { throw 'Enrichment failed' }
    .venv/Scripts/python.exe -m evaluation.runner $video --kind video --start 0 --stride 12 --mode VLM_PLUS_FIFTH_LAYER --detector-model yolo11n.pt --detector-device cuda --detector-confidence 0.4 --vlm-model $vlmSnapshot --vlm-device cuda --output "results_case_${case}_combined_sensory_raw.json"
    if ($LASTEXITCODE -ne 0) { throw 'Harness run failed' }
    .venv/Scripts/python.exe -m evaluation.sensory "results_case_${case}_combined_sensory_raw.json" --output "results_case_${case}_combined_sensory.json"
    if ($LASTEXITCODE -ne 0) { throw 'Enrichment failed' }
}
```

Add the original `--stop` if a baseline used a restricted range. Compare frame
numbers/source hashes with baseline manifests before interpreting results. The
enrichment CLI accepts only a new `*_sensory.json` output and opens it exclusively.
The source JSON is read only. It does not require models or re-decoding videos.

## Validation and working-tree scope

311 isolated unit/regression tests passed: 272 existing and 39 new. Breakdown:
66 world-model v0.1, 46 v0.2, 24 existing harness, 136 other existing regressions,
34 new sensory tests, 5 new enrichment tests. Tests cover all epistemic categories,
missing-versus-negative evidence, source score preservation, source time/identity,
provenance, deterministic permutations, physical constraints, explicit opposition,
actor safeguards, unchanged hypotheses without usable sensory evidence, real
harness mode separation, JSON round trips and overwrite protection. Missing
reasoners are tested as unavailable, not pretended to emit inferred consequences.

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$testModules = Get-ChildItem -Filter 'test_*.py' | Where-Object { (Get-Content -LiteralPath $_.FullName -Raw) -match 'unittest|def test_' } | ForEach-Object { $_.BaseName }
.venv/Scripts/python.exe -m unittest @testModules evaluation.tests.test_harness evaluation.tests.test_sensory -q
```

Eight deterministic legacy smoke scripts also exited successfully:
`test_active_perception.py`, `test_auditory.py`, `test_auditory_negative.py`,
`test_occlusion_reasoner.py`, `test_orchestrator.py`, `test_semantic_conflict.py`,
`test_temporal.py`, `test_yolo_occlusion.py`. These are not assertion-based test
counts. Five model/service smoke scripts were intentionally not run:
`test_active_perception_loop.py`, `test_locate_adapter.py`,
`test_perception_fusion.py`, `test_smolvlm_core.py`, `test_yolo_detector.py`.
They launch external services or neural inference at import time. Thus the full
isolated regression suite passed; unrestricted collection of every script was
not performed under the user's no-model-download/no-expensive-evaluation rules.

Modified: `fifth_layer/world_model/{__init__,evidence,evidence_providers,hypothesis_generator}.py`.
Created: `fifth_layer/world_model/sensory_evidence.py`, this document,
`test_sensory_evidence.py`, `evaluation/sensory.py`, `evaluation/tests/test_sensory.py`.
No other tracked files changed. `git diff --check` passed.

`live_app.py`, SmolVLM, detector/tracking, all existing reasoners, orchestrator,
and existing evaluation implementations/tests were untouched. No models downloaded,
no training, no learning, no external world models, no commit/push/tag operations.
HEAD remains `2b0634d`. The six baseline JSONs and `test_videos/` were already
untracked before this task and remain so. Git status comprises four modified
tracked files, five new implementation/test/documentation files, and those
pre-existing untracked inputs/results.

The six frozen JSONs have identical pre/post SHA-256 values:

| File | SHA-256 |
| --- | --- |
| results_case_a_combined.json | A2C478A99C107C95EC07975517F8DED58963C8CDDBEE0052166B01C29141DE7D |
| results_case_a_fifth_layer.json | 1D0866C30DC4E98B2CC17DB25EE40D7829C4354058D7D87886B693D1C15D0759 |
| results_case_a_vlm.json | 8C137B8E1F61C1F855F4FA0B33CFE2670620C35C323C60CA89DDDDB62EDA3F9E |
| results_case_b_combined.json | 955F4886D11987B08D06742038676C3B70705A256DF7C17ADABCAC62E6267B06 |
| results_case_b_fifth_layer.json | EA8FFC93F80C1EEA4C18FC78A0EE6BFF2AFE39DC6B4FB78B68AE552D14891C0E |
| results_case_b_vlm.json | EE8882BF018788613E185F902D451EFCBA5A4AB704AF2E0133259EFC3297E000 |
