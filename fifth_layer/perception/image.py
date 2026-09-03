"""Image perception module for Fifth Layer Engine v0.8."""

from pathlib import Path
from time import time

from PIL import Image

from fifth_layer.perception.base import BasePerception
from fifth_layer.world_state import WorldState


class ImagePerception(BasePerception):
    """Convert a local image into a simple WorldState."""

    def perceive(self, source) -> WorldState:
        image_path = Path(source)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image file not found: {image_path}"
            )

        with Image.open(image_path) as image:
            width, height = image.size
            image_format = image.format
            image_mode = image.mode

        return WorldState(
            timestamp=time(),
            data={
                "source_type": "image",
                "source_path": str(image_path),
                "image_width": width,
                "image_height": height,
                "image_format": image_format,
                "image_mode": image_mode,
            },
        )