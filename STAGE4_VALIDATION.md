# Stage 4: session prediction confidence calibration

Only finalized evaluable position forecasts train calibration. Correct has
weight 1, partially_correct 0.5, incorrect 0; unevaluable and expired are ignored.
Pending predictions do not contribute. No model training or saved learning state
is introduced. Existing diagnostic logging is retained.

For each bucket, reliability R = (2 + weighted_success)/(4 + total_evaluable).
Select the first bucket with at least five samples:
source + prediction_type + class_name -> source + prediction_type -> source.
If none qualifies, use the fixed global Beta(2,2) prior, R=0.5, with no confidence
adjustment. Source-level sample count is still shown while collecting evidence.
The prior never pools outcomes from different sources.

New confidence C = clip(raw * (1 + 0.20 * (2*R - 1)), 0, 1).
Thus the maximum effect is relative +/-20%, not +/-20 percentage points.
For raw=0.72, five correct evaluations produce R=7/9 and C=0.80. One subsequent
incorrect evaluation gives R=7/10 and C=0.7776. Sub-five groups use parent evidence
only when that parent itself has at least five evaluable samples.

Raw confidence is never overwritten. TemporalPredictionReasoner now exposes the
existing live UI policy explicitly: 0.65 for visibility/exit predictions, otherwise
0.50. Occlusion position estimates have a separate named raw prior of 0.50, not
YOLO confidence. Legacy prediction records without a raw field use 0.50 as an
explicit compatibility default. Types are trajectory_position and
occlusion_position: these are position-evaluation reliability estimates, not
proof that a predicted semantic occlusion event occurred.

Each issued evaluation forecast freezes raw_confidence, calibration_reliability,
calibrated_confidence, calibration_samples and calibration_bucket. Evaluation
updates statistics for future forecasts only. The CAL overlay shows the last
evaluated forecast's issuance-time fields. Its sample count may therefore lag
the current bucket by one or more results; it does not retrofit the past forecast.
Below five samples: CAL: collecting evidence | n=3/5. Otherwise it shows raw and
calibrated values. OpenCV's text renderer uses ASCII -> for the arrow.

Calibration is owned by PredictionEvaluationMemory and reset with that session.
Both observation and overlay evaluation paths update calibration upon finalization.
Memory limits: 256 buckets (least recently updated eviction), 4096 processed IDs,
300 s inactivity TTL for buckets and 300 s retention for processed evaluation IDs.
Observation/overlay ticks clean idle records too. A monotonic timestamp watermark
rejects replay after ID eviction/TTL. At ID saturation, unseen outcomes at or before
the evicted timestamp may also be dropped conservatively; they are never counted
twice. Bucket TTL resets an inactive aggregate; it is not a per-sample rolling TTL.

CONFIDENCE_CALIBRATION logs the evaluated ID, source, type, class, status, frozen
issuance fields, and bucket statistics (counts, weighted success, reliability,
last updated timestamp). No persistent calibration state is restored at startup.

WorldState remains backward compatible; evaluation_predictions holds optional
confidence fields available to future consumers. Live/deep reasoning display
metadata also carries issuance confidence when a forecast is admitted. Sensor
fusion and FifthLayerEngine are inspected but unchanged: fusion does not consume
the new position reliability as hidden-actor evidence. Existing confidence-only
inputs retain their behavior. Detector confidence, matching costs, identity
lifetime, prediction geometry and temporal_deep UI timeout remain unchanged.

Validation: 83 unittest cases pass across calibration, evaluation, tracking,
predicted tracks, camera integration, analysis connections and observation lifetime.
The temporal, orchestrator and occlusion executable examples also complete.
The tests include prior/fallback, source isolation, bounded relative changes,
partial weighting, duplicate/replay protection, TTL/capacity, reset, immutable
past records, future confidence changes, actual CAL drawing, legacy fusion,
and unchanged YOLO/tracker values. These examples are not counted as unittest cases.

Camera validation (not yet performed):

1. Start `.venv\Scripts\python.exe live_app.py` and open live camera. Move one
   object steadily; wait for EVAL and CAL collecting evidence.
2. Accumulate at least five evaluable results for temporal_live. Inspect
   CONFIDENCE_CALIBRATION counts; subsequently issued/evaluated predictions should
   show raw and calibrated values. Correct results should raise the latter.
3. Change direction abruptly several times so observed comparisons are incorrect.
   Check gradual decreases on future forecasts, always within raw*0.8..raw*1.2
   and [0,1]. A single failure must not cause a large confidence collapse.
4. Hide the object beyond evaluation grace. Unevaluable/expired must not increase
   incorrect_count. Occlusion_track results belong to their own buckets.
5. Wait for deep results and verify temporal_deep does not inherit temporal_live
   errors. Compare logs by source, type, class and prediction_id.
6. Restart the camera session; CAL should collect evidence from the prior again.
   Check that YOLO labels, IDs, PREDICTED boxes and deep UI timeout behave as before.
