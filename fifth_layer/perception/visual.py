"""OpenCV-based visual perception for Fifth Layer Engine v0.9."""

from pathlib import Path
from time import time

import cv2

from fifth_layer.perception.base import BasePerception
from fifth_layer.world_state import WorldState


class VisualPerception(BasePerception):
    """Extract basic visual features from an image using OpenCV."""

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

        grayscale = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        mean_brightness = float(
            grayscale.mean()
        )

        edges = cv2.Canny(
            grayscale,
            100,
            200,
        )

        edge_pixels = int(
            (edges > 0).sum()
        )

        return WorldState(
            timestamp=time(),
            data={
                "source_type": "image",
                "source_path": str(image_path),
                "image_width": width,
                "image_height": height,
                "mean_brightness": round(
                    mean_brightness,
                    3,
                ),
                "edge_pixels": edge_pixels,
            },
        )