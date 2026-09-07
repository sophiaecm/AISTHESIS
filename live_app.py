import os
import tempfile
import threading
import time
import tkinter as tk
from tkinter import filedialog

from collections import Counter, deque

import cv2
from ultralytics import YOLO

from fifth_layer.world_state import WorldState
from fifth_layer.perception.smolvlm_scene import SmolVLMScenePerception
from fifth_layer.perception.perception_fusion import PerceptionFusion
from fifth_layer.perception.temporal import extract_motion_evidence
from fifth_layer.reasoners.orchestrator import AisthesisOrchestrator


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_PATH = "yolo11n.pt"
CONFIDENCE_THRESHOLD = 0.25
CAMERA_INDEX = 0

CAMERA_WARMUP_SECONDS = 6.0
DESCRIPTION_INTERVAL_SECONDS = 10.0

# Stabilization
DESCRIPTION_MIN_HOLD_SECONDS = 7.0
REASONING_MIN_HOLD_SECONDS = 4.0

MOTION_HISTORY_SIZE = 7
MOTION_MIN_VOTES = 4


# ---------------------------------------------------------
# Models
# ---------------------------------------------------------

print("Loading YOLO...")

yolo_model = YOLO(MODEL_PATH)

print("YOLO ready.")


smolvlm_model = None

smolvlm_lock = threading.Lock()
model_load_lock = threading.Lock()

perception_fusion = PerceptionFusion()
orchestrator = AisthesisOrchestrator()


# ---------------------------------------------------------
# Shared AISTHESIS state
# ---------------------------------------------------------

latest_description = "Waiting for scene description..."

latest_reasoning = {
    "latent": "insufficient_evidence",
    "prediction": "indeterminate",
    "risk": "UNKNOWN",
    "uncertainty": 1.0,
}

latest_motion_evidence = []
latest_motion_summary = "waiting_for_motion"

analysis_running = False


# ---------------------------------------------------------
# Stabilization state
# ---------------------------------------------------------

description_last_changed_at = 0.0
reasoning_last_changed_at = 0.0

motion_history = deque(
    maxlen=MOTION_HISTORY_SIZE
)


# ---------------------------------------------------------
# SmolVLM
# ---------------------------------------------------------

def load_smolvlm():
    global smolvlm_model

    if smolvlm_model is not None:
        return

    with model_load_lock:

        if smolvlm_model is not None:
            return

        print(
            "Loading SmolVLM2 on GPU..."
        )

        smolvlm_model = SmolVLMScenePerception(
            device="cuda",
        )

        print(
            "SmolVLM2 ready."
        )


# ---------------------------------------------------------
# YOLO
# ---------------------------------------------------------

def run_yolo(frame):

    results = yolo_model.predict(
        source=frame,
        conf=CONFIDENCE_THRESHOLD,
        device=0,
        verbose=False,
    )

    result = results[0]

    return (
        result.plot(),
        result,
    )


# ---------------------------------------------------------
# YOLO -> WorldState
# ---------------------------------------------------------

def build_yolo_world_state(
    frame,
    result,
    source_type="camera",
):

    detections = []

    boxes = result.boxes

    if boxes is not None:

        for index in range(
            len(boxes)
        ):

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

            class_name = (
                result.names[
                    class_id
                ]
            )

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


# ---------------------------------------------------------
# Description stabilization
# ---------------------------------------------------------

def _normalize_description(
    text,
):
    return {
        word.strip(
            ".,!?;:()[]{}\"'"
        ).lower()
        for word in text.split()
        if len(word) > 2
    }


def _description_similarity(
    first,
    second,
):
    first_words = (
        _normalize_description(
            first
        )
    )

    second_words = (
        _normalize_description(
            second
        )
    )

    if (
        not first_words
        or not second_words
    ):
        return 0.0

    intersection = len(
        first_words
        & second_words
    )

    union = len(
        first_words
        | second_words
    )

    if union == 0:
        return 0.0

    return (
        intersection
        / union
    )


def update_stable_description(
    candidate,
):
    global latest_description
    global description_last_changed_at

    candidate = (
        candidate or ""
    ).strip()

    if not candidate:
        return

    now = time.time()

    temporary_states = {
        "Waiting for scene description...",
        "Camera warming up...",
        "Waiting for scene analysis...",
        "Analyzing visible scene...",
    }

    if latest_description in temporary_states:

        latest_description = candidate
        description_last_changed_at = now

        return

    similarity = (
        _description_similarity(
            latest_description,
            candidate,
        )
    )

    # Very similar result:
    # keep the existing description.
    if similarity >= 0.82:
        return

    elapsed = (
        now
        - description_last_changed_at
    )

    if (
        elapsed
        >= DESCRIPTION_MIN_HOLD_SECONDS
    ):
        latest_description = candidate

        description_last_changed_at = now


# ---------------------------------------------------------
# Reasoning stabilization
# ---------------------------------------------------------

def update_stable_reasoning(
    candidate,
):
    global latest_reasoning
    global reasoning_last_changed_at

    if not candidate:
        return

    now = time.time()

    current_signature = (
        latest_reasoning.get(
            "latent"
        ),
        latest_reasoning.get(
            "prediction"
        ),
        latest_reasoning.get(
            "risk"
        ),
    )

    candidate_signature = (
        candidate.get(
            "latent"
        ),
        candidate.get(
            "prediction"
        ),
        candidate.get(
            "risk"
        ),
    )

    # Same logical result:
    # update uncertainty only.
    if (
        candidate_signature
        == current_signature
    ):

        latest_reasoning[
            "uncertainty"
        ] = candidate.get(
            "uncertainty",
            latest_reasoning.get(
                "uncertainty",
                1.0,
            ),
        )

        return

    elapsed = (
        now
        - reasoning_last_changed_at
    )

    if (
        reasoning_last_changed_at == 0.0
        or elapsed
        >= REASONING_MIN_HOLD_SECONDS
    ):

        latest_reasoning = dict(
            candidate
        )

        reasoning_last_changed_at = now


# ---------------------------------------------------------
# Motion smoothing
# ---------------------------------------------------------

def reset_motion_smoothing():

    global motion_history

    motion_history.clear()


def _motion_candidate_from_evidence(
    motion_evidence,
):

    moving_objects = [
        item
        for item in motion_evidence
        if item.get(
            "motion_state"
        )
        != "stationary"
    ]

    if moving_objects:

        strongest = max(
            moving_objects,
            key=lambda item: item.get(
                "normalized_motion",
                0.0,
            ),
        )

        return {
            "class_name": strongest.get(
                "class_name",
                "object",
            ),
            "motion_state": strongest.get(
                "motion_state",
                "moving",
            ),
            "speed": strongest.get(
                "speed_pixels_per_second"
            ),
        }

    if motion_evidence:

        first = motion_evidence[0]

        return {
            "class_name": first.get(
                "class_name",
                "object",
            ),
            "motion_state": "stationary",
            "speed": 0.0,
        }

    return {
        "class_name": None,
        "motion_state": "no_match",
        "speed": None,
    }


def update_smoothed_motion(
    motion_evidence,
):

    global latest_motion_summary
    global motion_history

    candidate = (
        _motion_candidate_from_evidence(
            motion_evidence
        )
    )

    motion_history.append(
        candidate
    )

    if len(
        motion_history
    ) < 3:
        return

    keys = []

    for item in motion_history:

        key = (
            item.get(
                "class_name"
            ),
            item.get(
                "motion_state"
            ),
        )

        keys.append(
            key
        )

    counts = Counter(
        keys
    )

    strongest_key, votes = (
        counts.most_common(
            1
        )[0]
    )

    required_votes = min(
        MOTION_MIN_VOTES,
        max(
            2,
            len(
                motion_history
            )
            // 2
            + 1,
        ),
    )

    if votes < required_votes:
        return

    class_name, motion_state = (
        strongest_key
    )

    matching = [
        item
        for item in motion_history
        if (
            item.get(
                "class_name"
            ),
            item.get(
                "motion_state"
            ),
        )
        == strongest_key
    ]

    speeds = [
        float(
            item["speed"]
        )
        for item in matching
        if item.get(
            "speed"
        )
        is not None
    ]

    if speeds:

        average_speed = (
            sum(speeds)
            / len(speeds)
        )

    else:

        average_speed = None

    if motion_state == "no_match":

        latest_motion_summary = (
            "no_temporal_match"
        )

        return

    if motion_state == "stationary":

        if class_name:

            latest_motion_summary = (
                f"{class_name} stationary"
            )

        else:

            latest_motion_summary = (
                "tracked_objects_stationary"
            )

        return

    if (
        class_name
        and average_speed
        is not None
    ):

        latest_motion_summary = (
            f"{class_name} "
            f"{motion_state} "
            f"({average_speed:.1f} px/s)"
        )

    elif class_name:

        latest_motion_summary = (
            f"{class_name} "
            f"{motion_state}"
        )


# ---------------------------------------------------------
# Temporal motion
# ---------------------------------------------------------

def calculate_temporal_motion(
    previous_detections,
    current_detections,
    image_width,
    image_height,
    previous_time,
    current_time,
):

    global latest_motion_evidence

    if (
        not previous_detections
        or not current_detections
        or previous_time is None
    ):

        latest_motion_evidence = []

        update_smoothed_motion(
            []
        )

        return []

    delta_time = (
        current_time
        - previous_time
    )

    if delta_time <= 0:
        return []

    motion_evidence = (
        extract_motion_evidence(
            previous_detections=(
                previous_detections
            ),
            current_detections=(
                current_detections
            ),
            image_width=image_width,
            image_height=image_height,
            delta_time=delta_time,
        )
    )

    latest_motion_evidence = (
        motion_evidence
    )

    update_smoothed_motion(
        motion_evidence
    )

    return motion_evidence


# ---------------------------------------------------------
# SmolVLM + Fusion + Fifth Layer
# ---------------------------------------------------------

def analyze_existing_yolo_result(
    frame,
    yolo_result,
    source_type="camera",
    motion_evidence=None,
):

    global latest_reasoning

    temp_path = None

    try:

        load_smolvlm()

        yolo_state = (
            build_yolo_world_state(
                frame,
                yolo_result,
                source_type,
            )
        )

        vlm_frame = frame.copy()

        height, width = (
            vlm_frame.shape[:2]
        )

        max_dimension = 640

        if (
            max(
                height,
                width,
            )
            > max_dimension
        ):

            scale = (
                max_dimension
                / max(
                    height,
                    width,
                )
            )

            new_width = max(
                1,
                int(
                    width
                    * scale
                ),
            )

            new_height = max(
                1,
                int(
                    height
                    * scale
                ),
            )

            vlm_frame = (
                cv2.resize(
                    vlm_frame,
                    (
                        new_width,
                        new_height,
                    ),
                    interpolation=(
                        cv2.INTER_AREA
                    ),
                )
            )

        temp_file = (
            tempfile.NamedTemporaryFile(
                suffix=".jpg",
                delete=False,
            )
        )

        temp_path = (
            temp_file.name
        )

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

        fused_state = (
            perception_fusion.fuse(
                yolo_state,
                smolvlm_state,
            )
        )

        # -------------------------------------------------
        # Temporal evidence
        # -------------------------------------------------

        fused_state.data[
            "motion_evidence"
        ] = (
            motion_evidence
            or []
        )

        moving_items = [
            item
            for item in (
                motion_evidence
                or []
            )
            if item.get(
                "motion_state"
            )
            != "stationary"
        ]

        fused_state.data[
            "moving_object_count"
        ] = len(
            moving_items
        )

        if moving_items:

            strongest_motion = max(
                moving_items,
                key=lambda item: item.get(
                    "normalized_motion",
                    0.0,
                ),
            )

            fused_state.data[
                "strongest_motion"
            ] = strongest_motion

            fused_state.data[
                "motion_detected"
            ] = True

        else:

            fused_state.data[
                "strongest_motion"
            ] = None

            fused_state.data[
                "motion_detected"
            ] = False

        reasoning = (
            orchestrator.analyze(
                fused_state
            )
        )

        # -------------------------------------------------
        # Stable description
        # -------------------------------------------------

        description = (
            fused_state.data.get(
                "scene_description",
                "",
            )
            .strip()
        )

        if description:

            update_stable_description(
                description
            )

        # -------------------------------------------------
        # Temporal reasoning
        # -------------------------------------------------

        temporal = reasoning.get(
            "temporal",
            {},
        )

        temporal_latent = (
            temporal.get(
                "latent",
                {},
            )
        )

        temporal_future = (
            temporal.get(
                "future",
                {},
            )
        )

        temporal_prediction = (
            temporal_future.get(
                "predicted_event"
            )
        )

        temporal_state = (
            temporal_latent.get(
                "latent_temporal_state"
            )
        )

        # -------------------------------------------------
        # Sensor fusion reasoning
        # -------------------------------------------------

        fusion = reasoning.get(
            "sensor_fusion",
            {},
        )

        fusion_expected = (
            fusion.get(
                "expected",
                {},
            )
        )

        fusion_latent = (
            fusion.get(
                "latent",
                {},
            )
        )

        fusion_future = (
            fusion.get(
                "future",
                {},
            )
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

        # -------------------------------------------------
        # Select visible prediction
        #
        # Real observed motion gets temporal prediction.
        # Hidden-actor prediction is used only when there
        # is explicit hidden-actor evidence.
        # -------------------------------------------------

        moving_now = any(
            item.get(
                "motion_state"
            )
            != "stationary"
            for item in (
                motion_evidence
                or []
            )
        )

        if (
            moving_now
            and temporal_prediction
            and temporal_prediction
            != "indeterminate"
        ):

            candidate_reasoning = {
                "latent": (
                    temporal_state
                    or
                    "motion_continuation_possible"
                ),
                "prediction": (
                    temporal_prediction
                ),
                "risk": "UNKNOWN",
                "uncertainty": 0.50,
            }

        elif active_sources > 0:

            candidate_reasoning = {
                "latent": (
                    fusion_latent.get(
                        "latent_hypothesis",
                        "insufficient_evidence",
                    )
                ),
                "prediction": (
                    fusion_future.get(
                        "predicted_event",
                        "indeterminate",
                    )
                ),
                "risk": (
                    fusion_future.get(
                        "risk_level",
                        "UNKNOWN",
                    )
                ),
                "uncertainty": (
                    uncertainty
                ),
            }

        elif (
            temporal_prediction
            and temporal_prediction
            != "indeterminate"
        ):

            candidate_reasoning = {
                "latent": (
                    temporal_state
                    or
                    "insufficient_temporal_evidence"
                ),
                "prediction": (
                    temporal_prediction
                ),
                "risk": "UNKNOWN",
                "uncertainty": 0.70,
            }

        else:

            candidate_reasoning = {
                "latent": (
                    "insufficient_evidence"
                ),
                "prediction": (
                    "indeterminate"
                ),
                "risk": "UNKNOWN",
                "uncertainty": 1.0,
            }

        update_stable_reasoning(
            candidate_reasoning
        )

    except Exception as exc:

        print(
            "AISTHESIS analysis error:",
            exc,
        )

    finally:

        if (
            temp_path
            and os.path.exists(
                temp_path
            )
        ):

            try:

                os.remove(
                    temp_path
                )

            except OSError:
                pass


# ---------------------------------------------------------
# Background analysis
# ---------------------------------------------------------

def analyze_background(
    frame,
    yolo_result,
    source_type="camera",
    motion_evidence=None,
):

    global analysis_running

    if analysis_running:
        return

    analysis_running = True

    frame_copy = frame.copy()

    motion_copy = list(
        motion_evidence
        or []
    )

    def worker():

        global analysis_running

        try:

            analyze_existing_yolo_result(
                frame_copy,
                yolo_result,
                source_type,
                motion_copy,
            )

        finally:

            analysis_running = False

    threading.Thread(
        target=worker,
        daemon=True,
    ).start()


# ---------------------------------------------------------
# Overlay
# ---------------------------------------------------------

def draw_overlay(
    frame,
):

    description = (
        latest_description
    )

    latent = (
        latest_reasoning.get(
            "latent",
            "N/A",
        )
    )

    prediction = (
        latest_reasoning.get(
            "prediction",
            "N/A",
        )
    )

    risk = (
        latest_reasoning.get(
            "risk",
            "UNKNOWN",
        )
    )

    uncertainty = float(
        latest_reasoning.get(
            "uncertainty",
            1.0,
        )
    )

    motion = (
        latest_motion_summary
    )

    max_chars = 65

    words = (
        description.split()
    )

    lines = []
    current_line = ""

    for word in words:

        candidate = (
            current_line
            + " "
            + word
        ).strip()

        if (
            len(candidate)
            <= max_chars
        ):

            current_line = (
                candidate
            )

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
        200
        + len(lines)
        * 25
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
        f"MOTION: {motion}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    y += 27

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


# ---------------------------------------------------------
# Live Camera
# ---------------------------------------------------------

def run_live_camera():

    global latest_description
    global latest_reasoning
    global latest_motion_evidence
    global latest_motion_summary

    global description_last_changed_at
    global reasoning_last_changed_at

    latest_description = (
        "Camera warming up..."
    )

    latest_reasoning = {
        "latent": "insufficient_evidence",
        "prediction": "indeterminate",
        "risk": "UNKNOWN",
        "uncertainty": 1.0,
    }

    latest_motion_evidence = []

    latest_motion_summary = (
        "collecting_temporal_evidence"
    )

    description_last_changed_at = (
        time.time()
    )

    reasoning_last_changed_at = 0.0

    reset_motion_smoothing()

    previous_detections = []
    previous_detection_time = None

    cap = cv2.VideoCapture(
        CAMERA_INDEX
    )

    if not cap.isOpened():

        print(
            "Could not open camera."
        )

        return

    camera_started_at = (
        time.time()
    )

    last_analysis = (
        camera_started_at
    )

    while True:

        success, frame = (
            cap.read()
        )

        if not success:
            break

        (
            annotated_frame,
            yolo_result,
        ) = run_yolo(
            frame
        )

        current_state = (
            build_yolo_world_state(
                frame,
                yolo_result,
                "camera",
            )
        )

        current_detections = (
            current_state.data.get(
                "detections",
                [],
            )
        )

        image_width = (
            current_state.data.get(
                "image_width",
                frame.shape[1],
            )
        )

        image_height = (
            current_state.data.get(
                "image_height",
                frame.shape[0],
            )
        )

        current_detection_time = (
            time.time()
        )

        motion_evidence = (
            calculate_temporal_motion(
                previous_detections,
                current_detections,
                image_width,
                image_height,
                previous_detection_time,
                current_detection_time,
            )
        )

        previous_detections = (
            current_detections
        )

        previous_detection_time = (
            current_detection_time
        )

        current_time = (
            time.time()
        )

        camera_age = (
            current_time
            - camera_started_at
        )

        if (
            camera_age
            >= CAMERA_WARMUP_SECONDS
            and
            current_time
            - last_analysis
            >= DESCRIPTION_INTERVAL_SECONDS
        ):

            # IMPORTANT:
            # Do NOT replace the old description with
            # "Analyzing..." here.
            #
            # The last stable description remains visible
            # until the new analysis is actually ready.

            analyze_background(
                frame,
                yolo_result,
                "camera",
                motion_evidence,
            )

            last_analysis = (
                current_time
            )

        annotated_frame = (
            draw_overlay(
                annotated_frame
            )
        )

        cv2.imshow(
            "AISTHESIS Live",
            annotated_frame,
        )

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key == ord("q"):
            break

    cap.release()

    cv2.destroyAllWindows()


# ---------------------------------------------------------
# Photo
# ---------------------------------------------------------

def open_photo():

    global latest_description
    global latest_reasoning
    global latest_motion_summary
    global description_last_changed_at
    global reasoning_last_changed_at

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

    latest_motion_summary = (
        "not_available_for_single_image"
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

    description_last_changed_at = 0.0
    reasoning_last_changed_at = 0.0

    (
        annotated_image,
        yolo_result,
    ) = run_yolo(
        image
    )

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
        [],
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


# ---------------------------------------------------------
# Video
# ---------------------------------------------------------

def open_video():

    global latest_description
    global latest_reasoning
    global latest_motion_evidence
    global latest_motion_summary
    global description_last_changed_at
    global reasoning_last_changed_at

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

    latest_motion_evidence = []

    latest_motion_summary = (
        "collecting_temporal_evidence"
    )

    description_last_changed_at = (
        time.time()
    )

    reasoning_last_changed_at = 0.0

    reset_motion_smoothing()

    previous_detections = []
    previous_detection_time = None

    video_started_at = (
        time.time()
    )

    last_analysis = (
        video_started_at
    )

    while True:

        success, frame = (
            cap.read()
        )

        if not success:
            break

        (
            annotated_frame,
            yolo_result,
        ) = run_yolo(
            frame
        )

        current_state = (
            build_yolo_world_state(
                frame,
                yolo_result,
                "video",
            )
        )

        current_detections = (
            current_state.data.get(
                "detections",
                [],
            )
        )

        current_detection_time = (
            time.time()
        )

        motion_evidence = (
            calculate_temporal_motion(
                previous_detections,
                current_detections,
                current_state.data.get(
                    "image_width",
                    frame.shape[1],
                ),
                current_state.data.get(
                    "image_height",
                    frame.shape[0],
                ),
                previous_detection_time,
                current_detection_time,
            )
        )

        previous_detections = (
            current_detections
        )

        previous_detection_time = (
            current_detection_time
        )

        current_time = (
            time.time()
        )

        if (
            current_time
            - last_analysis
            >= DESCRIPTION_INTERVAL_SECONDS
        ):

            analyze_background(
                frame,
                yolo_result,
                "video",
                motion_evidence,
            )

            last_analysis = (
                current_time
            )

        annotated_frame = (
            draw_overlay(
                annotated_frame
            )
        )

        cv2.imshow(
            "AISTHESIS Video",
            annotated_frame,
        )

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key == ord("q"):
            break

    cap.release()

    cv2.destroyAllWindows()


# ---------------------------------------------------------
# Main UI
# ---------------------------------------------------------

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