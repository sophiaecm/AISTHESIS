"""Run a bounded local camera/GPU smoke test without opening UI windows."""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import time
import cv2
import live_app as app


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    if not cap.isOpened():
        raise RuntimeError("Camera unavailable")
    previous, previous_time = [], None
    started = time.monotonic()
    frames = 0
    identities = set()
    motion_frames = 0
    snapshot = None
    try:
        while frames < 90 and time.monotonic() - started < 25:
            ok, frame = cap.read()
            timestamp = time.time()
            if not ok:
                raise RuntimeError("Camera frame read failed")
            _, result = app.run_yolo(frame)
            state = app.build_yolo_world_state(frame, result, timestamp=timestamp)
            detections = app.track_observation(state)
            identities.update(item["track_id"] for item in detections)
            motion = app.calculate_temporal_motion(previous, detections, frame.shape[1], frame.shape[0], previous_time, timestamp)
            motion_frames += bool(motion)
            app.update_live_temporal_prediction(state)
            app.draw_overlay(frame.copy())
            snapshot = app.AnalysisSnapshot.capture(frame, state, app.latest_stable_motion_evidence)
            previous, previous_time = detections, timestamp
            frames += 1
        print("CAMERA_GPU", dict(frames=frames, tracks=len(identities), motion_frames=motion_frames,
                                 feedback=list(app.prediction_feedback.history)), flush=True)
        app.load_smolvlm()
        app.latest_description = "Awaiting smoke analysis"
        app.analyze_existing_yolo_result(None, None, snapshot=snapshot)
        if app.latest_description == "Awaiting smoke analysis":
            raise RuntimeError("Scene analysis did not produce a description")
        print("SCENE_ANALYSIS_OK", flush=True)
    finally:
        cap.release()


if __name__ == "__main__":
    main()
