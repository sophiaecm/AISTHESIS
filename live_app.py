import os
import tempfile
import threading
import time
import tkinter as tk
from tkinter import filedialog

import cv2
from ultralytics import YOLO

from fifth_layer.world_state import WorldState
from fifth_layer.perception.smolvlm_scene import SmolVLMScenePerception
from fifth_layer.perception.perception_fusion import PerceptionFusion
from fifth_layer.reasoners.orchestrator import AisthesisOrchestrator


MODEL_PATH = "yolo11n.pt"
CONFIDENCE_THRESHOLD = 0.25
CAMERA_INDEX = 0

# Do not analyze the first dark/exposure-adjusting camera frames.
CAMERA_WARMUP_SECONDS = 6.0

# SmolVLM is intentionally not called continuously.
DESCRIPTION_INTERVAL_SECONDS = 10.0


print("Loading YOLO...")

yolo_model = YOLO(MODEL_PATH)

print("YOLO ready.")


smolvlm_model = None
smolvlm_lock = threading.Lock()
model_load_lock = threading.Lock()

perception_fusion = PerceptionFusion()
orchestrator = AisthesisOrchestrator()


latest_description = "Waiting for scene description..."

latest_reasoning = {
    "latent": "insufficient_evidence",
    "prediction": "indeterminate",
    "risk": "UNKNOWN",
    "uncertainty": 1.0,
}

analysis_running = False


def load_smolvlm():
    global smolvlm_model

    if smolvlm_model is not None:
        return

    with model_load_lock:
        if smolvlm_model is not None:
            return

        print("Loading SmolVLM2 on GPU...")

        smolvlm_model = SmolVLMScenePerception(
            device="cuda",
        )

        print("SmolVLM2 ready.")


def run_yolo(frame):
    results = yolo_model.predict(
        source=frame,
        conf=CONFIDENCE_THRESHOLD,
        device=0,
        verbose=False,
    )

    result = results[0]

    return result.plot(), result


def build_yolo_world_state(
    frame,
    result,
    source_type="camera",
):
    detections = []

    boxes = result.boxes

    if boxes is not None:
        for index in range(len(boxes)):
            box = boxes[index]

            xyxy = (
                box.xyxy[0]
                .detach()
                .cpu()
                .tolist()
            )

            confidence = float(
                box.conf[0]
                .detach()
                .cpu()
                .item()
            )

            class_id = int(
                box.cls[0]
                .detach()
                .cpu()
                .item()
            )

            class_name = result.names[
                class_id
            ]

            detections.append(
                {
                    "object_id": index,
                    "class_name": class_name,
                    "confidence": confidence,
                    "box_xyxy": xyxy,
                }
            )

    height, width = frame.shape[:2]

    return WorldState(
        timestamp=time.time(),
        data={
            "source_type": source_type,
            "image_width": width,
            "image_height": height,
            "detections": detections,
            "detection_count": len(
                detections
            ),
            "model": MODEL_PATH,
        },
    )


def analyze_existing_yolo_result(
    frame,
    yolo_result,
    source_type="camera",
):
    global latest_description
    global latest_reasoning

    temp_path = None

    try:
        load_smolvlm()

        yolo_state = build_yolo_world_state(
            frame,
            yolo_result,
            source_type,
        )

        # Reduce the image passed to the VLM.
        # YOLO still sees the original frame.
        vlm_frame = frame.copy()

        height, width = vlm_frame.shape[:2]

        max_dimension = 640

        if max(height, width) > max_dimension:
            scale = (
                max_dimension
                / max(height, width)
            )

            new_width = max(
                1,
                int(width * scale),
            )

            new_height = max(
                1,
                int(height * scale),
            )

            vlm_frame = cv2.resize(
                vlm_frame,
                (
                    new_width,
                    new_height,
                ),
                interpolation=cv2.INTER_AREA,
            )

        temp_file = tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False,
        )

        temp_path = temp_file.name
        temp_file.close()

        cv2.imwrite(
            temp_path,
            vlm_frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                85,
            ],
        )

        with smolvlm_lock:
            smolvlm_state = (
                smolvlm_model.perceive(
                    temp_path
                )
            )

        fused_state = perception_fusion.fuse(
            yolo_state,
            smolvlm_state,
        )

        reasoning = orchestrator.analyze(
            fused_state
        )

        description = fused_state.data.get(
            "scene_description",
            "",
        ).strip()

        if description:
            latest_description = description

        fusion = reasoning.get(
            "sensor_fusion",
            {},
        )

        fusion_expected = fusion.get(
            "expected",
            {},
        )

        fusion_latent = fusion.get(
            "latent",
            {},
        )

        fusion_future = fusion.get(
            "future",
            {},
        )

        active_sources = int(
            fusion_expected.get(
                "active_evidence_sources",
                0,
            )
        )

        uncertainty = float(
            fusion_expected.get(
                "fused_uncertainty",
                1.0,
            )
        )

        if active_sources == 0:
            latest_reasoning = {
                "latent": "insufficient_evidence",
                "prediction": "indeterminate",
                "risk": "UNKNOWN",
                "uncertainty": 1.0,
            }

        else:
            latest_reasoning = {
                "latent": fusion_latent.get(
                    "latent_hypothesis",
                    "insufficient_evidence",
                ),
                "prediction": fusion_future.get(
                    "predicted_event",
                    "indeterminate",
                ),
                "risk": fusion_future.get(
                    "risk_level",
                    "UNKNOWN",
                ),
                "uncertainty": uncertainty,
            }

    except Exception as exc:
        print(
            "AISTHESIS analysis error:",
            exc,
        )

        latest_reasoning = {
            "latent": "analysis_error",
            "prediction": "indeterminate",
            "risk": "UNKNOWN",
            "uncertainty": 1.0,
        }

    finally:
        if (
            temp_path
            and os.path.exists(temp_path)
        ):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def analyze_background(
    frame,
    yolo_result,
    source_type="camera",
):
    global analysis_running

    if analysis_running:
        return

    analysis_running = True

    frame_copy = frame.copy()

    def worker():
        global analysis_running

        try:
            analyze_existing_yolo_result(
                frame_copy,
                yolo_result,
                source_type,
            )

        finally:
            analysis_running = False

    threading.Thread(
        target=worker,
        daemon=True,
    ).start()


def draw_overlay(frame):
    description = latest_description

    latent = latest_reasoning.get(
        "latent",
        "N/A",
    )

    prediction = latest_reasoning.get(
        "prediction",
        "N/A",
    )

    risk = latest_reasoning.get(
        "risk",
        "UNKNOWN",
    )

    uncertainty = float(
        latest_reasoning.get(
            "uncertainty",
            1.0,
        )
    )

    max_chars = 65

    words = description.split()

    lines = []
    current_line = ""

    for word in words:
        candidate = (
            current_line
            + " "
            + word
        ).strip()

        if len(candidate) <= max_chars:
            current_line = candidate

        else:
            if current_line:
                lines.append(
                    current_line
                )

            current_line = word

    if current_line:
        lines.append(
            current_line
        )

    lines = lines[:4]

    panel_height = (
        170
        + len(lines) * 25
    )

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (
            frame.shape[1] - 10,
            panel_height,
        ),
        (0, 0, 0),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.60,
        frame,
        0.40,
        0,
        frame,
    )

    cv2.putText(
        frame,
        "AISTHESIS",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "PREDICTIVE PERCEPTION",
        (170, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    y = 68

    cv2.putText(
        frame,
        "SCENE",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    y += 25

    for line in lines:
        cv2.putText(
            frame,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        y += 23

    y += 5

    cv2.putText(
        frame,
        f"LATENT: {latent}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    y += 27

    cv2.putText(
        frame,
        f"PREDICTION: {prediction}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    y += 27

    cv2.putText(
        frame,
        f"RISK: {str(risk).upper()}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    y += 27

    cv2.putText(
        frame,
        f"UNCERTAINTY: {uncertainty:.2f}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return frame


def run_live_camera():
    global latest_description
    global latest_reasoning

    latest_description = (
        "Camera warming up..."
    )

    latest_reasoning = {
        "latent": "insufficient_evidence",
        "prediction": "indeterminate",
        "risk": "UNKNOWN",
        "uncertainty": 1.0,
    }

    cap = cv2.VideoCapture(
        CAMERA_INDEX
    )

    if not cap.isOpened():
        print(
            "Could not open camera."
        )
        return

    camera_started_at = time.time()

    last_analysis = camera_started_at

    while True:
        success, frame = cap.read()

        if not success:
            break

        # YOLO runs only once.
        annotated_frame, yolo_result = (
            run_yolo(frame)
        )

        current_time = time.time()

        camera_age = (
            current_time
            - camera_started_at
        )

        if (
            camera_age
            >= CAMERA_WARMUP_SECONDS
            and current_time
            - last_analysis
            >= DESCRIPTION_INTERVAL_SECONDS
        ):
            latest_description = (
                "Analyzing visible scene..."
            )

            analyze_background(
                frame,
                yolo_result,
                "camera",
            )

            last_analysis = current_time

        annotated_frame = draw_overlay(
            annotated_frame
        )

        cv2.imshow(
            "AISTHESIS Live",
            annotated_frame,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def open_photo():
    global latest_description
    global latest_reasoning

    file_path = (
        filedialog.askopenfilename(
            title="Choose Photo",
            filetypes=[
                (
                    "Image files",
                    "*.jpg *.jpeg *.png *.bmp *.webp",
                )
            ],
        )
    )

    if not file_path:
        return

    image = cv2.imread(
        file_path
    )

    if image is None:
        print(
            "Could not open image."
        )
        return

    annotated_image, yolo_result = (
        run_yolo(image)
    )

    latest_description = (
        "Analyzing visible scene..."
    )

    latest_reasoning = {
        "latent": "analyzing",
        "prediction": "analyzing",
        "risk": "UNKNOWN",
        "uncertainty": 1.0,
    }

    # Show immediately before deeper reasoning.
    preview = draw_overlay(
        annotated_image.copy()
    )

    cv2.imshow(
        "AISTHESIS Photo",
        preview,
    )

    cv2.waitKey(1)

    analyze_existing_yolo_result(
        image,
        yolo_result,
        "image",
    )

    final_image = draw_overlay(
        annotated_image.copy()
    )

    cv2.imshow(
        "AISTHESIS Photo",
        final_image,
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def open_video():
    global latest_description
    global latest_reasoning

    file_path = (
        filedialog.askopenfilename(
            title="Choose Video",
            filetypes=[
                (
                    "Video files",
                    "*.mp4 *.avi *.mov *.mkv *.webm",
                )
            ],
        )
    )

    if not file_path:
        return

    cap = cv2.VideoCapture(
        file_path
    )

    if not cap.isOpened():
        print(
            "Could not open video."
        )
        return

    latest_description = (
        "Waiting for scene analysis..."
    )

    latest_reasoning = {
        "latent": "insufficient_evidence",
        "prediction": "indeterminate",
        "risk": "UNKNOWN",
        "uncertainty": 1.0,
    }

    video_started_at = time.time()
    last_analysis = video_started_at

    while True:
        success, frame = cap.read()

        if not success:
            break

        annotated_frame, yolo_result = (
            run_yolo(frame)
        )

        current_time = time.time()

        if (
            current_time
            - last_analysis
            >= DESCRIPTION_INTERVAL_SECONDS
        ):
            analyze_background(
                frame,
                yolo_result,
                "video",
            )

            last_analysis = current_time

        annotated_frame = draw_overlay(
            annotated_frame
        )

        cv2.imshow(
            "AISTHESIS Video",
            annotated_frame,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    root = tk.Tk()

    root.title(
        "AISTHESIS"
    )

    root.geometry(
        "420x360"
    )

    title_label = tk.Label(
        root,
        text="AISTHESIS",
        font=(
            "Arial",
            24,
            "bold",
        ),
    )

    title_label.pack(
        pady=25
    )

    subtitle_label = tk.Label(
        root,
        text="Predictive Perception",
        font=(
            "Arial",
            12,
        ),
    )

    subtitle_label.pack(
        pady=5
    )

    live_button = tk.Button(
        root,
        text="Live Camera",
        width=24,
        height=2,
        command=run_live_camera,
    )

    live_button.pack(
        pady=10
    )

    photo_button = tk.Button(
        root,
        text="Open Photo",
        width=24,
        height=2,
        command=open_photo,
    )

    photo_button.pack(
        pady=10
    )

    video_button = tk.Button(
        root,
        text="Open Video",
        width=24,
        height=2,
        command=open_video,
    )

    video_button.pack(
        pady=10
    )

    root.mainloop()


if __name__ == "__main__":
    main()