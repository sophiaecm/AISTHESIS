import os
import json
import logging
from logging.handlers import RotatingFileHandler
import tempfile
import threading
import time
import tkinter as tk
from tkinter import filedialog

from collections import Counter, deque

import cv2
from ultralytics import YOLO

from fifth_layer.perception.live_inference import LiveInferenceWorker
from fifth_layer.world_state import WorldState
from fifth_layer.perception.analysis_snapshot import AnalysisSnapshot
from fifth_layer.perception.tracking import ObjectTracker
from fifth_layer.prediction_feedback import PredictionFeedback
from fifth_layer.prediction_evaluation import PredictionEvaluationMemory, evaluation_overlay
from fifth_layer.perception.smolvlm_scene import SmolVLMScenePerception
from fifth_layer.perception.perception_fusion import PerceptionFusion
from fifth_layer.perception.temporal import extract_motion_evidence
from fifth_layer.reasoners.orchestrator import AisthesisOrchestrator
from fifth_layer.reasoners.temporal_prediction import TemporalPredictionReasoner


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
PREDICTION_DISPLAY_HOLD_SECONDS = 2.0

MOTION_HISTORY_SIZE = 7
MOTION_MIN_VOTES = 4

# Live performance / reliability
LIVE_YOLO_EVERY_N_FRAMES = 2
LIVE_YOLO_EVERY_N_FRAMES_DURING_VLM = 4
LIVE_MIN_TRACK_CONFIDENCE = 0.65
LIVE_MAX_NORMALIZED_SPEED = 1.25
SMOLVLM_MAX_DIMENSION = 448


# ---------------------------------------------------------
# Models
# ---------------------------------------------------------

print("Loading YOLO...")

yolo_model = YOLO(MODEL_PATH)

print("YOLO ready.")


smolvlm_model = None

smolvlm_lock = threading.Lock()
model_load_lock = threading.Lock()
yolo_lock = threading.Lock()

perception_fusion = PerceptionFusion()
orchestrator = AisthesisOrchestrator()
temporal_reasoner = TemporalPredictionReasoner()
# Temporary diagnostic log: at most three 512 KiB files; no per-frame UI output.
track_logger = logging.getLogger("aisthesis.tracks")
track_logger.setLevel(logging.INFO)
track_logger.propagate = False
if not track_logger.handlers:
    track_handler = RotatingFileHandler(
        os.path.join(tempfile.gettempdir(), "aisthesis-tracks.log"),
        maxBytes=512 * 1024, backupCount=2, encoding="utf-8",
    )
    track_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    track_logger.addHandler(track_handler)


def log_track_event(event, **fields):
    track_logger.info("%s %s", event, json.dumps(fields, ensure_ascii=True))


object_tracker = ObjectTracker(
    preserve_observation_continuity=True, observation_expiration=True,
    min_confidence=LIVE_MIN_TRACK_CONFIDENCE, event_logger=log_track_event,
    predict_missing_tracks=True,
)
prediction_feedback = PredictionFeedback()
prediction_evaluations = PredictionEvaluationMemory(event_logger=log_track_event)


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

# UI-only cache. The inference state remains real-time, while the last
# meaningful temporal prediction stays visible briefly for readability.
display_reasoning = dict(latest_reasoning)
display_reasoning_last_valid_at = 0.0

latest_motion_evidence = []
latest_stable_motion_evidence = []
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

def infer_yolo(frame):
    # Serialize model access across live, photo and video sessions.
    with yolo_lock:
        return yolo_model.predict(
            source=frame, conf=CONFIDENCE_THRESHOLD, device=0, verbose=False,
        )[0]


def run_yolo(frame):
    result = infer_yolo(frame)
    return result.plot(), result


# ---------------------------------------------------------
# YOLO -> WorldState
# ---------------------------------------------------------

def build_yolo_world_state(
    frame,
    result,
    source_type="camera",
    timestamp=None,
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
        timestamp=time.time() if timestamp is None else timestamp,
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


def draw_tracked_detections(frame, detections):
    """Draw only actual detections, with session identities when available."""
    annotated = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection["box_xyxy"])
        identity = detection.get("track_id")
        suffix = f" #{identity}" if identity is not None else ""
        label = (f"{detection.get('class_name', 'unknown')}{suffix} "
                 f"{float(detection.get('confidence', 0.0)):.2f}")
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, label, (x1, max(15, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return annotated


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
    global latest_stable_motion_evidence

    object_tracker.reset()
    prediction_feedback.reset()
    prediction_evaluations.reset()
    motion_history.clear()
    latest_stable_motion_evidence = []


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
            "track_id": strongest.get("track_id"),
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
            "track_id": first.get("track_id"),
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
    global latest_stable_motion_evidence
    global motion_history

    candidate = (
        _motion_candidate_from_evidence(
            motion_evidence
        )
    )

    motion_history.append(
        candidate
    )

    # Until a short history exists, do not expose motion
    # to the predictive reasoner.
    if len(motion_history) < 3:
        latest_motion_summary = "collecting_temporal_evidence"
        latest_stable_motion_evidence = []
        return

    keys = [
        (
            item.get("class_name"),
            item.get("motion_state"),
            item.get("track_id"),
        )
        for item in motion_history
    ]

    strongest_key, votes = (
        Counter(keys).most_common(1)[0]
    )

    required_votes = min(
        MOTION_MIN_VOTES,
        max(
            2,
            len(motion_history) // 2 + 1,
        ),
    )

    if votes < required_votes:
        latest_motion_summary = "motion_unconfirmed"
        latest_stable_motion_evidence = []
        return

    class_name, motion_state, track_id = strongest_key

    matching_history = [
        item
        for item in motion_history
        if (
            item.get("class_name"),
            item.get("motion_state"),
            item.get("track_id"),
        ) == strongest_key
    ]

    speeds = [
        float(item["speed"])
        for item in matching_history
        if item.get("speed") is not None
    ]

    average_speed = (
        sum(speeds) / len(speeds)
        if speeds
        else None
    )

    if motion_state == "no_match":
        latest_stable_motion_evidence = []
        latest_motion_summary = "no_temporal_match"
        return

    if motion_state == "stationary":
        latest_stable_motion_evidence = []
        latest_motion_summary = (
            f"{class_name} stationary"
            if class_name
            else "tracked_objects_stationary"
        )
        return

    # Preserve geometric fields from the newest raw evidence
    # that agrees with the winning stabilized state.
    raw_matches = [
        item
        for item in motion_evidence
        if item.get("class_name") == class_name
        and item.get("motion_state") == motion_state
        and item.get("track_id") == track_id
    ]

    if not raw_matches:
        latest_motion_summary = "motion_unconfirmed"
        latest_stable_motion_evidence = []
        return

    strongest_raw = max(
        raw_matches,
        key=lambda item: item.get(
            "normalized_motion",
            0.0,
        ),
    )

    stable_item = dict(strongest_raw)

    raw_speed = stable_item.get(
        "speed_pixels_per_second"
    )

    if (
        average_speed is not None
        and raw_speed is not None
        and float(raw_speed) > 0.0
    ):
        scale = average_speed / float(raw_speed)

        if stable_item.get("velocity_x") is not None:
            stable_item["velocity_x"] = (
                float(stable_item["velocity_x"])
                * scale
            )

        if stable_item.get("velocity_y") is not None:
            stable_item["velocity_y"] = (
                float(stable_item["velocity_y"])
                * scale
            )

        stable_item[
            "speed_pixels_per_second"
        ] = average_speed

    latest_stable_motion_evidence = [
        stable_item
    ]

    if class_name and average_speed is not None:
        latest_motion_summary = (
            f"{class_name} {motion_state} "
            f"({average_speed:.1f} px/s)"
        )
    elif class_name:
        latest_motion_summary = (
            f"{class_name} {motion_state}"
        )


# ---------------------------------------------------------
# Temporal motion
# ---------------------------------------------------------

def filter_live_tracking_detections(
    detections,
):
    """Keep only detections reliable enough to drive live motion reasoning.

    Low-confidence one-frame class hallucinations may still be drawn by YOLO,
    but they do not become temporal evidence.
    """

    return [
        detection
        for detection in detections
        if float(
            detection.get(
                "confidence",
                0.0,
            )
        )
        >= LIVE_MIN_TRACK_CONFIDENCE
    ]


def track_observation(current_state):
    data = current_state.data
    visible = object_tracker.update(
        data.get("detections", []),
        current_state.timestamp, data["image_width"], data["image_height"],
    )
    data["detections"] = visible
    data["accepted_detections"] = visible
    data["detection_count"] = len(visible)
    data["predicted_tracks"] = list(object_tracker.predicted_tracks)
    data["prediction_evaluations"] = prediction_evaluations.observe(
        current_state.timestamp, visible, data["image_width"], data["image_height"])
    data["evaluation_predictions"] = prediction_evaluations.record_state(current_state, {}, "occlusion_track")
    visible = filter_live_tracking_detections(visible)
    data["prediction_feedback"] = data["prediction_evaluations"]
    return visible


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
            minimum_motion_pixels=8.0,
            minimum_normalized_motion=0.01,
        )
    )

    latest_motion_evidence = (
        motion_evidence
    )

    update_smoothed_motion(
        motion_evidence
    )

    return motion_evidence


def update_live_temporal_prediction(
    current_state,
):
    """Run lightweight trajectory reasoning every frame.

    SmolVLM is intentionally not involved here. This keeps
    motion -> trajectory -> latent -> future prediction live.

    Temporal predictions are cancelled as soon as stabilized
    motion disappears, so an old trajectory cannot remain visible
    after the tracked object becomes stationary.
    """

    global latest_reasoning
    global reasoning_last_changed_at

    stable_motion = list(
        latest_stable_motion_evidence
    )

    moving_items = [
        item
        for item in stable_motion
        if item.get(
            "motion_state"
        )
        != "stationary"
    ]

    # -----------------------------------------------------
    # Temporal prediction expiration / cancellation
    # -----------------------------------------------------
    #
    # A trajectory prediction is only valid while stabilized
    # motion evidence still supports it. If the object stops,
    # immediately remove a previous temporal-live prediction.
    #
    # Do not erase a sensor-fusion result here.
    # -----------------------------------------------------

    if not moving_items:

        if (
            latest_reasoning.get("source")
            == "temporal_live"
        ):
            latest_reasoning = {
                "latent": "insufficient_evidence",
                "prediction": "indeterminate",
                "risk": "UNKNOWN",
                "uncertainty": 1.0,
                "source": "temporal_expired",
            }

            reasoning_last_changed_at = (
                time.time()
            )

        return

    temporal_data = dict(
        current_state.data
    )

    temporal_data[
        "motion_evidence"
    ] = moving_items

    temporal_data[
        "moving_object_count"
    ] = len(
        moving_items
    )

    temporal_data[
        "strongest_motion"
    ] = max(
        moving_items,
        key=lambda item: item.get(
            "normalized_motion",
            0.0,
        ),
    )

    temporal_data[
        "motion_detected"
    ] = True

    temporal_state = WorldState(
        timestamp=current_state.timestamp,
        data=temporal_data,
    )

    expected = (
        temporal_reasoner
        .infer_expected_consequences(
            temporal_state
        )
    )

    latent = (
        temporal_reasoner
        .infer_latent_state(
            temporal_state,
            expected,
        )
    )

    future = (
        temporal_reasoner
        .infer_future_state(
            latent
        )
    )

    current_state.data.setdefault("evaluation_predictions", []).extend(
        prediction_evaluations.record_state(temporal_state, expected.predictions, "temporal_live"))

    latent_name = latent.features.get(
        "latent_temporal_state",
        "motion_continuation_possible",
    )

    prediction = future.data.get(
        "predicted_event",
        "indeterminate",
    )

    if prediction in {
        None,
        "indeterminate",
        "tracked_objects_likely_remain_stationary",
    }:

        if (
            latest_reasoning.get("source")
            == "temporal_live"
        ):
            latest_reasoning = {
                "latent": "insufficient_evidence",
                "prediction": "indeterminate",
                "risk": "UNKNOWN",
                "uncertainty": 1.0,
                "source": "temporal_expired",
            }

            reasoning_last_changed_at = (
                time.time()
            )

        return

    occlusion_states = {
        "trajectory_toward_occlusion",
        "visibility_loss_possible",
        "trajectory_toward_view_exit",
    }

    uncertainty = (
        0.35
        if latent_name in occlusion_states
        else 0.50
    )

    latest_reasoning = {
        "latent": latent_name,
        "prediction": prediction,
        "risk": "UNKNOWN",
        "uncertainty": uncertainty,
        "source": "temporal_live",
    }

    reasoning_last_changed_at = (
        time.time()
    )


# ---------------------------------------------------------
# SmolVLM + Fusion + Fifth Layer
# ---------------------------------------------------------

def analyze_existing_yolo_result(
    frame,
    yolo_result,
    source_type="camera",
    motion_evidence=None,
    snapshot=None,
):

    global latest_reasoning

    temp_path = None

    try:

        if snapshot is None:
            snapshot = AnalysisSnapshot.capture(
                frame,
                build_yolo_world_state(frame, yolo_result, source_type),
                motion_evidence or [],
            )
        yolo_state = snapshot.world_state()
        stable_motion_evidence = snapshot.motion_evidence()
        vlm_frame = snapshot.frame().copy()
        load_smolvlm()

        height, width = (
            vlm_frame.shape[:2]
        )

        max_dimension = SMOLVLM_MAX_DIMENSION

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

        fused_state.timestamp = snapshot.timestamp
        fused_state.data["observation_timestamp"] = snapshot.timestamp

        fused_state.data[
            "motion_evidence"
        ] = stable_motion_evidence

        moving_items = [
            item
            for item in stable_motion_evidence
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

        prediction_evaluations.record_state(fused_state, temporal.get("expected", {}), "temporal_deep")

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
        # Priority:
        # 1. Occlusion-aware temporal prediction
        # 2. Other temporal motion prediction
        # 3. Supported sensor-fusion prediction
        # 4. Indeterminate
        #
        # Important:
        # Generic occlusion is not treated as proof of a
        # hidden actor. These states describe visibility
        # changes of an already observed tracked object.
        # -------------------------------------------------

        moving_now = any(
            item.get(
                "motion_state"
            )
            != "stationary"
            for item in stable_motion_evidence
        )

        occlusion_temporal_states = {
            "trajectory_toward_occlusion",
            "visibility_loss_possible",
            "trajectory_toward_view_exit",
        }

        if (
            moving_now
            and temporal_state
            in occlusion_temporal_states
            and temporal_prediction
            and temporal_prediction
            != "indeterminate"
        ):

            candidate_reasoning = {
                "latent": temporal_state,
                "prediction": temporal_prediction,
                "risk": "UNKNOWN",
                "uncertainty": 0.35,
                "source": "temporal_deep",
            }

        elif (
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
                "prediction": temporal_prediction,
                "risk": "UNKNOWN",
                "uncertainty": 0.50,
                "source": "temporal_deep",
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
                "uncertainty": uncertainty,
                "source": "sensor_fusion",
            }

        else:

            candidate_reasoning = {
                "latent": "insufficient_evidence",
                "prediction": "indeterminate",
                "risk": "UNKNOWN",
                "uncertainty": 1.0,
                "source": "insufficient_evidence",
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
    world_state=None,
):

    global analysis_running

    if analysis_running:
        return

    snapshot = AnalysisSnapshot.capture(
        frame,
        world_state if world_state is not None else
        build_yolo_world_state(frame, yolo_result, source_type),
        motion_evidence or [],
    )
    analysis_running = True

    def worker():

        global analysis_running

        try:

            analyze_existing_yolo_result(
                None,
                None,
                source_type,
                snapshot=snapshot,
            )

        finally:

            analysis_running = False

    threading.Thread(
        target=worker,
        daemon=True,
    ).start()



def get_display_reasoning():
    """Return a readable UI view without delaying inference-state changes."""

    global display_reasoning
    global display_reasoning_last_valid_at

    now = time.time()

    current = dict(
        latest_reasoning
    )

    prediction = current.get(
        "prediction",
        "indeterminate",
    )

    meaningful = prediction not in {
        None,
        "",
        "indeterminate",
        "analyzing",
    }

    if meaningful:
        display_reasoning = current
        display_reasoning_last_valid_at = now
        return display_reasoning

    if (
        display_reasoning_last_valid_at > 0.0
        and now - display_reasoning_last_valid_at
        < PREDICTION_DISPLAY_HOLD_SECONDS
    ):
        return display_reasoning

    display_reasoning = current
    return display_reasoning


# ---------------------------------------------------------
# Overlay
# ---------------------------------------------------------

def draw_overlay(
    frame,
    detections=None,
    predicted_tracks=None,
):

    description = (
        latest_description
    )

    ui_reasoning = (
        get_display_reasoning()
    )

    latent = (
        ui_reasoning.get(
            "latent",
            "N/A",
        )
    )

    prediction = (
        ui_reasoning.get(
            "prediction",
            "N/A",
        )
    )

    risk = (
        ui_reasoning.get(
            "risk",
            "UNKNOWN",
        )
    )

    uncertainty = float(
        ui_reasoning.get(
            "uncertainty",
            1.0,
        )
    )

    motion = (
        latest_motion_summary
    )
    evaluation_line = evaluation_overlay(prediction_evaluations.latest(time.time()))


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
        227
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

    if evaluation_line:
        cv2.putText(frame, evaluation_line, (20, y + 27), cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, (255, 255, 255), 1, cv2.LINE_AA)

    for prediction in predicted_tracks or []:
        if time.time() > prediction["prediction_valid_until"]:
            continue
        x1, y1, x2, y2 = map(int, prediction["predicted_bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 1)
        cv2.putText(frame, f"PREDICTED {prediction['class_name']} #{prediction['track_id']}",
                    (x1, max(15, y1-7)), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 165, 255), 1)
    if detections is not None:
        frame = draw_tracked_detections(frame, detections)
    return frame


# ---------------------------------------------------------
# Live Camera
# ---------------------------------------------------------

def run_live_camera():

    global latest_description
    global latest_reasoning
    global latest_motion_evidence
    global latest_stable_motion_evidence
    global latest_motion_summary
    global display_reasoning
    global display_reasoning_last_valid_at

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

    display_reasoning = dict(
        latest_reasoning
    )
    display_reasoning_last_valid_at = 0.0

    latest_motion_evidence = []
    latest_stable_motion_evidence = []

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

    # Keep the capture buffer short when the backend supports it.
    # This reduces the feeling of watching old queued frames.
    try:
        cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1,
        )
    except Exception:
        pass

    camera_started_at = (
        time.time()
    )

    last_analysis = (
        camera_started_at
    )

    frame_index = 0
    current_state = None
    inference = LiveInferenceWorker(infer_yolo)
    try:
        while True:
            success, frame = cap.read()
            frame_timestamp = time.time()
            if not success:
                break
            frame_index += 1
            completed = inference.poll()
            if completed is not None:
                observed_frame, observed_at, yolo_result, error = completed
                if error is not None:
                    print("Live YOLO error:", error)
                else:
                    current_state = build_yolo_world_state(
                        observed_frame, yolo_result, "camera", timestamp=observed_at,
                    )
                    current_detections = track_observation(current_state)
                    calculate_temporal_motion(
                        previous_detections, current_detections,
                        current_state.data["image_width"], current_state.data["image_height"],
                        previous_detection_time, observed_at,
                    )
                    previous_detections = current_detections
                    previous_detection_time = observed_at
                    update_live_temporal_prediction(current_state)
                    now = time.time()
                    if (not analysis_running
                            and now - camera_started_at >= CAMERA_WARMUP_SECONDS
                            and now - last_analysis >= DESCRIPTION_INTERVAL_SECONDS):
                        analyze_background(
                            observed_frame, yolo_result, "camera",
                            list(latest_stable_motion_evidence), world_state=current_state,
                        )
                        last_analysis = now

            stride = (LIVE_YOLO_EVERY_N_FRAMES_DURING_VLM if analysis_running
                      else LIVE_YOLO_EVERY_N_FRAMES)
            if current_state is None or frame_index % stride == 0:
                inference.submit(frame, frame_timestamp)

            # Reuse actual last-observed boxes between completed inferences.
            # This display cache never updates tracking or predicts missing boxes.
            detections = current_state.data["detections"] if current_state else []
            annotated_frame = draw_overlay(frame.copy(), detections,
                                           current_state.data.get("predicted_tracks", []) if current_state else [])
            cv2.imshow("AISTHESIS Live", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        inference.close()
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
    global latest_stable_motion_evidence
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
    latest_stable_motion_evidence = []

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
        frame_timestamp = time.time()

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
                timestamp=frame_timestamp,
            )
        )

        current_detections = (
            track_observation(current_state)
        )

        annotated_frame = frame.copy()

        current_detection_time = current_state.timestamp

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

        update_live_temporal_prediction(current_state)

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
                list(latest_stable_motion_evidence),
                world_state=current_state,
            )

            last_analysis = (
                current_time
            )

        annotated_frame = (
            draw_overlay(
                annotated_frame, current_state.data["detections"], current_state.data.get("predicted_tracks", [])
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
