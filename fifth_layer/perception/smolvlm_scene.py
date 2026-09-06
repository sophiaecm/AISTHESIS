from pathlib import Path
from time import time

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

from fifth_layer.perception.base import BasePerception
from fifth_layer.world_state import WorldState


class SmolVLMScenePerception(BasePerception):
    """Generate visible-scene evidence using SmolVLM2."""

    def __init__(
        self,
        model_id="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        device="cuda",
    ):
        self.model_id = model_id

        if device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.float16
        else:
            self.device = "cpu"
            self.dtype = torch.float32

        print(f"Loading SmolVLM2 on {self.device}...")

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
        )

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
        ).to(self.device)

        self.model.eval()

        print("SmolVLM2 loaded.")

    def perceive(self, source) -> WorldState:
        image_path = Path(source)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image file not found: {image_path}"
            )

        image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {
                        "type": "text",
                        "text": (
                            "Analyze only visible evidence in the image. "
                            "Describe the scene and identify the main visible "
                            "objects, people, object types, and spatial relations. "
                            "Be specific about unusual objects when possible. "
                            "Do not infer hidden or unseen objects."
                        ),
                    },
                ],
            }
        ]

        prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=prompt,
            images=[image],
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            if hasattr(value, "to")
            else value
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
            )

        input_length = inputs["input_ids"].shape[1]

        generated_only = generated_ids[:, input_length:]

        description = self.processor.batch_decode(
            generated_only,
            skip_special_tokens=True,
        )[0].strip()

        return WorldState(
            timestamp=time(),
            data={
                "source_type": "image",
                "source_path": str(image_path),
                "perception_module": "smolvlm2",
                "model": self.model_id,
                "device": self.device,
                "scene_description": description,
            },
        )