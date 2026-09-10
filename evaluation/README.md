# Offline evaluation foundation

This namespace measures mode separation and inspectable output contributions.
It does not implement Experiment 001, score ground truth, or claim that richer
output is better. All production files, world-model contracts and algorithms
remain unchanged. Importing evaluation does not load a model.

## Run with existing local models

From the repository root, using the existing project environment:

```powershell
.venv/Scripts/python.exe -m evaluation.runner input.jpg --vlm-model C:/path/to/existing/SmolVLM/snapshot --detector-model yolo11n.pt --output evaluation-result.json
.venv/Scripts/python.exe -m evaluation.runner input.mp4 --kind video --start 0 --stop 100 --stride 5 --vlm-model C:/path/to/existing/SmolVLM/snapshot --output video-result.json
.venv/Scripts/python.exe -m evaluation.runner input.mp4 --kind video --stop 30 --mode FIFTH_LAYER_ONLY --output baseline-result.json
```

The output path must not already exist. Detector weights must already exist as
a local file, and VLM weights must be an existing local snapshot directory.
Workers set HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE before importing models.
No model download or network service is part of the harness. Missing local
dependencies/models produce failed-worker diagnostics, never a substituted model.
CPU is the default evaluation device; explicit --detector-device / --vlm-device
options select existing implementation devices. Detection threshold defaults
to YoloDetectorPerception's 0.40; configuration is recorded for each run.

## Exact mode paths

VLM_ONLY: prepared image -> unchanged SmolVLMScenePerception.perceive -> raw
WorldState JSON. No detector, tracker, fusion or Fifth Layer reasoning call.
The existing prompt and generation behavior are not replaced. Metadata records
the perceive method source, exact string constants (including prompt), model
directory, effective device, configuration and installed library versions.

FIFTH_LAYER_ONLY: prepared image -> unchanged YoloDetectorPerception -> strict
detector schema audit -> fresh TrackMemory -> existing geometry/motion extraction
-> strict reasoner-input audit -> AisthesisOrchestrator.analyze. It never calls
SmolVLM or PerceptionFusion.fuse. Allowed inputs are exhaustively:

- image_width and image_height from the detector;
- detections: detector class_id/class_name/confidence and box_xyxy (legacy box
  and object_id also accepted by the explicit detector schema);
- accepted_detections, detection_count and predicted_tracks from fresh tracking;
- motion_evidence from extract_motion_evidence using consecutive selected frames;
- scene_relations from PerceptionFusion._build_scene_relations, a geometry-only
  helper; this does not call the fusion pipeline;
- occlusion_evidence from extract_occlusion_evidence;
- WorldState.timestamp from the selected frame's evaluation timeline.

There are no supplied VLM labels/text, cached language outputs, physics inputs,
world-model hypotheses, auditory inputs, or previous production-session state.
Detector metadata remains in raw output/configuration, outside reasoner inputs.
Tracking uses the repository's legacy defaults, including missing-track
prediction disabled; it resets per condition and persists only across selected
frames in that condition. The first selected frame has no prior motion evidence.
Large sampling strides can expire tracks under those unchanged defaults.

VLM_PLUS_FIFTH_LAYER: the same detector/tracking/geometry path plus unchanged
SmolVLM -> PerceptionFusion.fuse -> AisthesisOrchestrator.analyze. Observation
timestamp and motion summaries are carried into fusion's returned WorldState,
as in the repository's existing orchestration. The JSON separately retains the
raw detector output, structured state, raw VLM output, exact reasoner input,
complete orchestrator output, and its sensor_fusion branch as fusion_output.
This is the repository's modular combined path, not a reproduction of live UI
scheduling, image downscaling, timeouts, calibration, or display selection.
No active-perception search/model is invoked in response to a generated query.

The orchestrator still runs its unchanged semantic-conflict logic in the
baseline. With an absent description, it can report unsupported-description
conflicts and active-perception requests. These are source behavior, not evidence
of VLM leakage or measured usefulness. They are preserved rather than suppressed.

## Same input and isolation

Every comparison uses one source hash and one frame manifest. An image is copied
byte-for-byte to a temporary snapshot. Video is decoded sequentially once into
temporary lossless PNGs. All modes read identical prepared bytes, verified
before/after calls, with source path/hash, selected frame number, sequence and
timestamp recorded. No live camera or persistent media copies are required.
Temporary media/control files are cleaned when the run exits.

Video selection is start-inclusive, stop-exclusive and uses the specified stride.
Timestamps are frame_number/source_FPS; they are **not VFR presentation timestamps**.
Use constant-frame-rate controlled videos for timing experiments with this version.
The exact range actually decoded is reflected in the manifest/results.

The runner launches a fresh Python process for each mode, with fresh adapters,
tracker, models and reasoner instances. Mode order cannot share Python globals
or model objects. Each result identifies the worker PID and isolation mechanism.
The injected-adapter execute API is useful for tests but explicitly labels its
default isolation as unverified. Models are independently run per condition;
do_sample=False is used by existing SmolVLM, but GPU/library nondeterminism and
timing differences remain possible. No seed or production behavior is changed.

## Leakage audit

Checks include:

- allowlisted detector root/detection fields before structured processing;
- allowlisted Fifth Layer input fields and recursive forbidden-origin markers;
- actual function arguments and their serialized JSON representation;
- nested dataclass/snapshot, provenance and JSON-encoded payload markers;
- adapter Python state/caches and loaded fifth_layer module mutable globals;
- presence of the SmolVLM module in a baseline process;
- shared-state checks both before and after reasoning;
- source/prepared-media hashes and production implementation hashes.

Unexpected data is rejected before reasoning, with raw detector output and
findings retained in JSON. Late shared-state findings mark output rejected.
The status `no_markers_in_tested_boundary` describes only these checks. It does
not prove universal absence of leakage. In particular:

- arbitrary detector class text cannot reveal its true origin from syntax alone;
- opaque neural/native state, function closures, external services, GPU and OS
  caches are not exhaustively inspected;
- serialized opaque formats and paraphrased/disguised content may evade markers;
- configured model paths are metadata, excluded from evidence-state scans;
- changing the adapters/custom injected code changes the trust boundary.

Tests deliberately inject description fields, nested serialized semantic data,
provenance markers, instance caches and module-global caches. These are rejected.
No real-model leakage conclusion has been measured in this implementation task.

## JSON and contribution comparison

The run envelope includes schema_version, run_id, config, environment/library
versions, preparation timing, production source hashes, worker logs/exit codes,
per-frame results and comparisons. Failed workers stay visible; incomplete mode
sets are not marked comparable. No failed mode is silently counted as a success.

Each result contains input identity, mode/status, configuration, perception,
vlm_output, fifth_layer_input, fifth_layer_output, fusion_output, provenance,
timing, leakage_diagnostics and claims. Raw output fields and original source
timestamps remain intact. Ground_truth=null and experiment_annotations={} are
extension slots only; they compute no metrics or calibration.

Claims are exact case-folded, whitespace-normalized string leaves and explicit
boolean atoms, retaining component and JSON field path. Numeric geometry and
metadata are excluded from this textual comparison but remain in raw outputs.
No LLM, embedding, synonym inference or semantic judge is used. Whole VLM text
can therefore fail to match an equivalent structured Fifth Layer label.

Fifth/fusion claims that also occur in their supplied inputs are marked
also_in_input and excluded from additional-inference comparison. This guards
against counting the reasoners' copied WorldState fields as new contribution;
it is conservative and can also exclude genuinely recomputed identical facts.
Comparisons expose shared, vlm_only, fifth_layer_only and combined_only atoms.
These are mode-specific differences, not causal attribution or quality judgments.

## Timing and Experiment 001

perf_counter measures perception (detector + structured processing), VLM,
Fifth Layer, fusion, and total per-frame mode latency. Lazy model loading is
included in first-call latency; no warm-up results are silently discarded.
Envelope timing separately records preparation and worker wall time, including
process startup. No latency benchmark was performed with real models here.

The harness accepts future ball-to-human and ball-to-no-human offline videos
with identical selection policies. It is ready to retain annotations for future
hidden-actor/event ground truth, Time-to-Anticipation, error rates, sensory
ablations and calibration. None of those experiments/metrics, video generation,
learning, neural training or world-model integration is implemented.

## Tests

```powershell
.venv/Scripts/python.exe -m unittest evaluation.tests.test_harness -q
```

Tests mock neural calls but exercise real repository structured perception,
fusion and reasoning. Tiny synthetic media are temporary. A fresh-process test
runs all conditions with fixture model adapters and checks distinct worker PIDs.
