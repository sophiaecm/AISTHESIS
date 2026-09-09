# Stage 3: prediction evaluation

Measurement only: no confidence calibration, online learning, reasoner weights,
or temporal_deep UI timeout changes. Legacy PredictionFeedback remains available
for existing callers; live measurement uses PredictionEvaluationMemory.

Temporal trajectories retain the originating WorldState timestamp and track ID.
Deep results use the captured snapshot and retained observed history, never the
current frame's geometry. Occlusion estimates are issued for their current target
time (zero forward horizon), and checked against a subsequent real observation.
Their origin center/bbox remains the last observed geometry. Predictions do not
enter detections or ground truth. Missing track IDs are never matched by class.

The symmetric target window is +/-0.25 s. Finalization waits until target+0.25 s
while the camera is running, not until the application window closes. Each new
YOLO result calls track_observation -> observe -> advance. Every overlay draw
also calls latest -> advance, so missing evidence can finalize without a new
YOLO result. The camera shutdown block performs resource cleanup only.
Finalization
and chooses the closest available same-track observation, including a pre-target
observation only if it follows prediction creation. Results finalize once.
Absent evidence is unevaluable; observations lost to the 60 s retention window
produce expired. Changed image geometry is not scored. Late detector results
arriving after finalization do not revise results; sparse/slow inference can
therefore produce unevaluable. This is not a position prediction failure.

Named thresholds: correct requires center error / image diagonal <=0.05 and
direction agreement. Partial requires <=0.12 or IoU>=0.30. Otherwise an observed
comparison is incorrect. At 640x480 these center limits are 40/96 pixels, and at
1280x720 approximately 73.4/176.2 pixels. These are broad initial image-relative
tolerances, not object-relative precision claims. Stationary displacement is
<=1% of the image diagonal; both movements inside that band agree without an
angle check. Otherwise directions must be within 45 degrees. One stationary
and one moving displacement do not agree.

Pending forecasts, retained individual observations, and results are each capped
at 512 records. Observations/results have a 60 s TTL; forecast horizon admission
is bounded by TTL minus grace. Cleanup runs during observations and overlay
ticks. When pending storage is full, new forecasts are not admitted. No future
forecast is prematurely finalized to free capacity. Reset clears session memory.
Admission allows one pending forecast per track/source/horizon. Source reports
are available through results_for_source('temporal_live'|'temporal_deep'|
'occlusion_track'). Detailed immutable copies of forecast, observation, metrics,
and timestamps are emitted as PREDICTION_EVALUATION using the existing logger.

FifthLayerEngine.compare keeps its original dictionary comparison. Its optional
prediction/observation arguments use the same compare_prediction helper as the
memory. Time-window scheduling belongs to memory; the engine does not schedule
or feed results back into reasoning. WorldState uses optional data keys only.

Camera check:

1. Run `.venv\Scripts\python.exe live_app.py` and select live camera normally.
2. Move steadily with one visible track. After the horizon and grace interval,
   check the single EVAL line and PREDICTION_EVALUATION log timestamps/track ID.
3. Change direction abruptly; look for partial/incorrect observed comparisons.
4. Hide briefly, then return with the same ID. Check occlusion_track records;
   hide longer than grace and verify unevaluable instead of incorrect.
5. Repeat with two same-class objects. Verify observations never cross IDs.
6. Allow a deep analysis to finish after movement. Verify temporal_deep target
   times still reference the old snapshot and appear in separate source reports.
7. Restart the session and verify old results disappear. Check the overlay stays
   at one EVAL line and predictions/confidence behavior remains as before.

Automated checks: test_prediction_evaluation covers statuses, identity, observed
filter, target/grace/nearest selection, finalization, normalization, stationary
and moving direction, delayed deep results, sources, TTL/capacity, legacy engine
and WorldState, and the actual live overlay through the existing AST test harness.

Validation result: 66 unittest cases passed in the project .venv. The live-loop
regression uses the real tracker, evaluation memory, track_observation and
draw_overlay functions with mocked camera/inference I/O. For correct, partial
and incorrect outcomes, it verifies finalization on an observed result at
target+0.25, the EVAL text before q/shutdown, and persistence on the next loop
iteration without duplicate evaluation. Existing
temporal, orchestrator, occlusion reasoner and YOLO occlusion executable examples
completed successfully. The real YOLO/SmolVLM perception-fusion smoke script also
completed successfully (it contains no unittest assertions). Camera validation
is still manual and has not been performed for this stage.
