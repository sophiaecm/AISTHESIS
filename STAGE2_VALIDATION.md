# Stage 2 live validation notes

Stage 1 was accepted in live camera testing. A single startup #0 -> #1
transition remains a known observation; this change does not attempt to fix it.

Observed detections remain in detections/accepted_detections. predicted_tracks
contains only estimates without detector confidence. position_uncertainty is
an increasing pixel uncertainty heuristic, not a calibrated probability.

Prediction horizons: normal 1.5 seconds, possible overlap/occlusion 2.5 seconds,
measured from the last accepted observation. The center is the raw constant
velocity estimate; the display bbox is shifted inside image bounds, retaining
its size unless larger than the image. Unreliable velocity freezes position.

Only completed YOLO observations update predictions/miss counts. The overlay
may hide an expired cached estimate between observations but never extrapolates
on skipped frames. Identity memory can outlive the displayed position estimate.
Reappearance error is reported only on the current matched detection and only
while the previous estimate remains within its validity horizon.

Manual check: move steadily, briefly hide behind another object, then reappear.
Look for a thin orange PREDICTED label and the same ID after reappearance.
Repeat while stationary, near image edges and with two people. Normal estimates
vanish after 1.5 s; possible occlusion estimates after at most 2.5 s. Real
observed labels remain green. No predicted object should increase detection_count.
