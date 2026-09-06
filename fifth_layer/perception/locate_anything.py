import json
import subprocess
import tempfile
from pathlib import Path
from time import time

from PIL import Image

from fifth_layer.perception.base import BasePerception
from fifth_layer.world_state import WorldState


class LocateAnythingPerception(BasePerception):
    """
    Event-triggered LocateAnything adapter with a persistent worker process.
    """

    def __init__(
        self,
        worker_script=r"C:\Users\ecmtu\AISTHESIS-Core\locate_worker_server.py",
        python_executable=r"C:\Users\ecmtu\AISTHESIS\.venv-locate\Scripts\python.exe",
        max_image_size=640,
    ):
        self.worker_script = Path(worker_script)
        self.python_executable = Path(python_executable)
        self.max_image_size = max_image_size
        self.process = None

    def _start_worker(self):
        if self.process is not None and self.process.poll() is None:
            return

        self.process = subprocess.Popen(
            [
                str(self.python_executable),
                str(self.worker_script),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        while True:
            line = self.process.stdout.readline()

            if not line:
                raise RuntimeError(
                    "LocateAnything worker stopped before READY."
                )

            line = line.strip()

            print(line)

            if line == "READY":
                break

    def _prepare_image(self, source):
        source_path = Path(source).resolve()

        if not source_path.exists():
            raise FileNotFoundError(
                f"Image file not found: {source_path}"
            )

        image = Image.open(source_path).convert("RGB")

        original_width, original_height = image.size

        image.thumbnail(
            (self.max_image_size, self.max_image_size)
        )

        resized_width, resized_height = image.size

        temp_file = tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False,
        )

        temp_path = Path(temp_file.name)
        temp_file.close()

        image.save(
            temp_path,
            format="JPEG",
            quality=85,
            optimize=True,
        )

        return (
            source_path,
            temp_path,
            original_width,
            original_height,
            resized_width,
            resized_height,
        )

    def perceive(
        self,
        source,
        targets=None,
        trigger_reason="manual_query",
    ) -> WorldState:

        if not targets:
            raise ValueError(
                "LocateAnything requires specific targets."
            )

        self._start_worker()

        (
            source_path,
            temp_path,
            original_width,
            original_height,
            resized_width,
            resized_height,
        ) = self._prepare_image(source)

        request = {
            "image_path": str(temp_path),
            "targets": targets,
        }

        try:
            self.process.stdin.write(
                json.dumps(request) + "\n"
            )
            self.process.stdin.flush()

            while True:
                line = self.process.stdout.readline()

                if not line:
                    raise RuntimeError(
                        "LocateAnything worker stopped unexpectedly."
                    )

                line = line.strip()

                try:
                    response = json.loads(line)
                    break
                except json.JSONDecodeError:
                    print(line)

            if not response.get("ok"):
                raise RuntimeError(
                    response.get(
                        "error",
                        "Unknown LocateAnything error",
                    )
                )

            result = response.get("result")

        finally:
            if temp_path.exists():
                temp_path.unlink()

        return WorldState(
            timestamp=time(),
            data={
                "source_type": "image",
                "source_path": str(source_path),
                "perception_module": "locate_anything",
                "execution_mode": "persistent_worker",
                "trigger_reason": trigger_reason,
                "targets": targets,
                "original_image_size": [
                    original_width,
                    original_height,
                ],
                "inference_image_size": [
                    resized_width,
                    resized_height,
                ],
                "result": result,
            },
        )

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()

            self.process = None