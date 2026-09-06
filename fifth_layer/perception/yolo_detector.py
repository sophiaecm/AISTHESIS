from pathlib import Path
from time import time

from ultralytics import YOLO

from fifth_layer.perception.base import BasePerception
from fifth_layer.world_state import WorldState


class YoloDetectorPerception(BasePerception):
    """Run Ultralytics YOLO and return detections as a Fifth Layer WorldState."""

    def __init__(
        self,
        model_path="yolo11n.pt",
        confidence_threshold=0.40,
        device=0,
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device

        self.model = YOLO(self.model_path)

    def perceive(self, source) -> WorldState:
        image_path = Path(source)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image file not found: {image_path}"
            )

        results = self.model.predict(
            source=str(image_path),
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )

        result = results[0]

        detections = []

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())

                xyxy = box.xyxy[0].tolist()

                x1, y1, x2, y2 = [
                    round(float(value), 1)
                    for value in xyxy
                ]

                class_name = self.model.names.get(
                    class_id,
                    "unknown",
                )

                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": round(confidence, 3),
                        "box_xyxy": [x1, y1, x2, y2],
                    }
                )

        height, width = result.orig_shape

        return WorldState(
            timestamp=time(),
            data={
                "source_type": "image",
                "source_path": str(image_path),
                "image_width": width,
                "image_height": height,
                "detector": "ultralytics_yolo",
                "model": str(self.model_path),
                "detections": detections,
                "detection_count": len(detections),
            },
        )