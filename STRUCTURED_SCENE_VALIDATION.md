# Structured Scene Narrator

## Stabilization update

Fast publication now uses FAST_SCENE_MIN_DISPLAY_SECONDS=3.0 and
FAST_SCENE_NORMAL_UPDATE_INTERVAL_SECONDS=1.5. Normal changes require two
consecutive YOLO observations with the same semantic signature. The stricter
3-second display minimum wins. This replaces the original 0.3-second debounce.

Signature: accepted class/count and image zones, categorical motion, predicted
and occlusion state, latent/future enum, risk and significant uncertainty bands.
It excludes raw confidence, exact pixels, frame age, timestamps, latency and
evaluation/calibration counters. Acceptance-threshold crossings still affect
counts, but normal count changes require confirmation.

Critical risk entry, actor emergence/collision/immediate-hazard events, newly
moving road actors and observed-to-predicted/occlusion transitions bypass holds.
Existing narrator wording is unchanged, including its supported event vocabulary.

Inference gaps no longer expire the displayed text automatically. Retained text
is last-known narration, not fresh evidence. Clearing requires two actual
observations without accepted/predicted objects or active/temporarily-missing
track memory, plus the minimum display time. Incoming observation freshness
admission remains 3 seconds. Reset clears state. Track memory is read-only.
UPDATED events occur only when displayed text changes; frame rendering continues.

Stabilization verification: 136 tests passed, including confidence/bbox jitter,
direction confirmation, minimum hold, dropped detection, track-memory retention,
immediate critical changes and existing fast/deep independence regressions.
The original implementation notes below describe the initial version; the timing
and expiry rules in this update supersede those original defaults.

StructuredSceneNarrator is deterministic English text composition over existing
WorldState/dictionary fields, without models, downloads or external calls.
FastSceneState retains one result, observes a 3 s freshness/display lifetime,
and debounces changed descriptions for 0.3 s. Increasing risk bypasses debounce,
including when the observation timestamp is unchanged. Identical text refreshes
freshness without changing wording. No narration runs in draw_overlay.

Live camera and video call update_fast_scene after completed YOLO tracking and
stable motion update. The adapter passes accepted_detections, predicted_tracks,
stable_motion_evidence and current temporal_live reasoning on a shallow copy;
deep reasoning is not borrowed from an old frame. Missing optional evidence is
reported as unavailable rather than invented. Old WorldState remains supported;
legacy detections without explicit accepted_detections are not asserted visible.

Evidence: accepted detection confidence >=0.5, class, track/object IDs,
box_xyxy/legacy box, image dimensions, scene_relations, stable motion or explicitly
stable motion_evidence, predicted track state/validity, occlusion_evidence,
latent/predicted_event, risk/uncertainty and calibrated confidence. The established
spatial helpers also derive one bounded near/overlap relation when none is supplied.
All spatial language is image-relative, never physical depth or distance to user.
Motion smaller than normalized 0.01 is not narrated as movement. Unknown future
event strings are not blindly converted into claims.

Observed language: 'One person is visible...'. Inferred: 'The movement pattern
suggests...'. Predicted: 'A previously observed ... may ...; its position is an
estimate.' Uncertain/rejected/predicted detections never enter visible counts.
No clothing, gender, age, color, intention or semantic scene details are inferred.

Modes: access defaults to accessibility_detailed, prioritizing dynamic objects
and understandable image positions. ADAS sorts road actors ahead of decorative
objects and defaults to detailed; callers can request concise for future speech
consumers. No speech engine is introduced. High/immediate risk always leads.
The live application explicitly chooses detailed/access.

Levels: concise <=3 sentences; detailed <=8; accessibility_detailed <=10.
Defaults allow 1600 characters. Both character and sentence limits are configurable
through the constructor. Class counts are grouped, only three class descriptions
are expanded and the remainder is summarized. Exact duplicate sentences are
removed. Summary fields retain additional evidence omitted from the main prose.

Output includes description, detail_level, mode, objects/spatial/motion/occlusion/
latent/prediction/risk/uncertainty summaries, evidence_types, generated_timestamp,
source=structured_local and latency_ms. User text excludes track IDs; structured
objects_summary retains them. Full output is state.data['fast_scene_narration'].

Live state exposes separate fast_scene_description and deep_scene_description,
fast/deep timestamps, fast latency, detail level and mode. FAST STRUCTURED SCENE
and DEEP VISUAL DESCRIPTION are separate overlay sections. To preserve the
existing panel, fast preview is capped at five wrapped lines (ellipsis when
truncated), deep at three. Full detailed output remains in WorldState. Smaller
than normal camera resolutions may need a larger window to read the panel.
Deep readiness, timeout and scene TTL never clear fast state. Fast TTL does not
clear deep. No existing model, tracker, evaluation, calibration or deep timeout
formula/threshold is modified.

FAST_SCENE_GENERATED/UPDATED/UNCHANGED/ERROR go to the existing log through a
64-item nonblocking queue and one daemon consumer. Overflow drops diagnostics,
never waits in the event loop. Fields include observation timestamp, mode, level,
latency, accepted/predicted counts, sentence count and update reason.

Validation: 120 tests passed, including existing scene/deep lifetime, tracking,
evaluation and calibration tests. Narration-only benchmark, 100 accepted objects
over 100 detailed runs: mean 0.305 ms, maximum 1.538 ms on this machine. No YOLO,
VLM, logging or UI time is included. This is a local test measurement, not a
hard real-time guarantee. Camera validation has not been performed.

Manual validation:

1. Run `.venv\Scripts\python.exe live_app.py`, open live camera. FAST should
   populate after the first accepted observation, before SmolVLM completes.
2. Move a person left/right and hold still. Verify stabilized motion and image
   regions, without jitter claims or appearance details in FAST.
3. Hide briefly. Previously observed estimates must not increase visible counts.
4. Wait for SmolVLM: DEEP updates separately and can contain visual details.
5. Verify deep timeout/TTL leaves FAST intact. Stop fresh YOLO observations:
   after 3 s FAST should show the waiting placeholder instead of stale prose.
6. Inspect aisthesis-tracks.log FAST_SCENE events for latency/update reasons.
   Full narration can be inspected in current_state.data['fast_scene_narration'].

No commit or push was performed.
