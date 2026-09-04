"""ONNX object detection for Fifth Layer Engine v0.10."""

from pathlib import Path
from time import time

import cv2
import numpy as np

from fifth_layer.perception.base import BasePerception
from fifth_layer.world_state import WorldState


COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


class ObjectDetectionPerception(BasePerception):
    """Run local ONNX object detection using OpenCV DNN."""

    def __init__(
        self,
        model_path="models/yolo11n.onnx",
        confidence_threshold=0.40,
        nms_threshold=0.45,
    ):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}"
            )

        self.net = cv2.dnn.readNetFromONNX(
            str(self.model_path)
        )

    def perceive(self, source) -> WorldState:
        image_path = Path(source)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image file not found: {image_path}"
            )

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(
                f"Could not read image: {image_path}"
            )

        height, width = image.shape[:2]

        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1 / 255.0,
            size=(640, 640),
            swapRB=True,
            crop=False,
        )

        self.net.setInput(blob)
        output = self.net.forward()

        detections = self._parse_output(
            output,
            width,
            height,
        )

        return WorldState(
            timestamp=time(),
            data={
                "source_type": "image",
                "source_path": str(image_path),
                "image_width": width,
                "image_height": height,
                "detections": detections,
                "detection_count": len(detections),
            },
        )

    def _parse_output(
        self,
        output,
        image_width,
        image_height,
    ):
        boxes = []
        confidences = []
        class_ids = []

        predictions = np.squeeze(output)

        if predictions.ndim != 2:
            return []

        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T

        x_factor = image_width / 640
        y_factor = image_height / 640

        for row in predictions:
            class_scores = row[4:]

            if len(class_scores) == 0:
                continue

            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])

            if confidence < self.confidence_threshold:
                continue

            x_center, y_center, box_width, box_height = row[:4]

            left = int(
                (x_center - box_width / 2) * x_factor
            )
            top = int(
                (y_center - box_height / 2) * y_factor
            )
            width = int(box_width * x_factor)
            height = int(box_height * y_factor)

            boxes.append(
                [left, top, width, height]
            )
            confidences.append(confidence)
            class_ids.append(class_id)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            self.confidence_threshold,
            self.nms_threshold,
        )

        detections = []

        for index in indices:
            i = int(
                np.asarray(index).reshape(-1)[0]
            )

            class_id = class_ids[i]

            class_name = (
                COCO_CLASSES[class_id]
                if 0 <= class_id < len(COCO_CLASSES)
                else "unknown"
            )

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(
                        confidences[i],
                        3,
                    ),
                    "box": boxes[i],
                }
            )

        return detections