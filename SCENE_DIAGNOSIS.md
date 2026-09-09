# Scene regression diagnosis and measured correction

The stage-5 publication callback contained update_stable_description as well as
temporal evaluation and reasoning publication. Deep timeout/stale rejection skipped
the whole callback, including scene. Deep display TTL did not clear scene text.
The configured temporal max age was 15 seconds, not 5; display TTL was 5 seconds.

Existing Windows temp rotating logs show workers starting (sequences 4-12), then
TIMED_OUT around 12.016-12.047 seconds. They did not record inference completion,
model load duration or exception tracebacks, so past successful inference versus
exceptions cannot be reconstructed conclusively from them alone.

Local offline measurements on this machine, using the existing cached
SmolVLM2-500M-Video-Instruct model on CUDA, no downloads:

| Input | Model load | Inference | Output |
|---|---:|---:|---|
| Existing test.jpg, original dimensions | 3.844 s | 20.141 s | Nonempty scene description |
| Same image, 448x280, JPEG quality 85 as in live preprocessing | 3.844 s | 19.625 s | Nonempty scene description |

These are controlled still-image measurements, not live camera frame benchmarks.
The 12-second temporal timeout is shorter than both measured inference runs.
Transformers printed configuration/deprecation warnings but both runs completed
without exception. No model or generation setting was changed.

Correction: scene is admitted immediately after successful SmolVLM inference,
before temporal fusion/publication. Its independent maximum snapshot age is 45 s,
display TTL 25 s. This permits measured cold load+inference (~23.5 s) and retention
while waiting for the next ~20 s run. Display also stops at the snapshot age cap.
These are initial measured-machine limits, not a claim of instantaneous scene
awareness. Temporal timeout/max age/display TTL remain 12/15/5 seconds.

The variable is latest_description, not latest_scene_description. Historically it
was written by update_stable_description, with a similarity >=0.82 suppression and
a minimum hold interval, and replaced with placeholders at mode startup. Camera
scene now lives in DeepAnalysisState.scene; overlay reads scene_description().
This prevents description smoothing from refreshing the lifetime of older text.
The legacy latest_description path remains for synchronous photos. Expired camera
scene returns Analyzing scene..., never the stale legacy global. Scene reset/close
clears scene state; temporal TTL clears only temporal state. Stale/empty scene
results preserve a still-valid previous scene until its own TTL expires.

The overlay always draws the SCENE area and wraps at most four description lines.
Previously, rejected publication left the startup placeholder unchanged; there
was no separate condition intentionally hiding SCENE when deep was stale.

SCENE and TEMPORAL_DEEP events are now mirrored to terminal by a filtered stream
handler (AISTHESIS_SCENE_DEBUG=1, default). Set it to 0 to disable routine console
diagnostics. The existing rotating file still receives events. Errors always
print SCENE_ANALYSIS_ERROR plus traceback, and are also logged. New events expose
start, model load duration, inference duration, output presence, completion,
snapshot age, independent scene/temporal acceptance and exact rejection reason.
Timing uses the existing monotonic controller clock. Terminal traces from the
next camera run provide actual live timings rather than inferring them from tests.

102 automated tests passed, including slow scene acceptance with rejected
temporal prediction, independent TTLs, old-scene suppression, full worker flow,
visible exception traceback, event loop/shutdown, and unchanged tracking,
evaluation and calibration regressions. Camera hardware validation remains manual.
