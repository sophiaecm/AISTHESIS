import json
import sys

sys.path.insert(
    0,
    r"C:\Users\ecmtu\AISTHESIS\Eagle\Embodied",
)

import torch
from PIL import Image
from locateanything_worker import LocateAnythingWorker


MODEL_ID = "nvidia/LocateAnything-3B"


def main():
    print("Loading LocateAnything...")
    sys.stdout.flush()

    worker = LocateAnythingWorker(
        MODEL_ID,
        device="cpu",
        dtype=torch.float16,
        use_batch_runtime=False,
    )

    print("READY")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()

        if not line:
            continue

        try:
            request = json.loads(line)

            image_path = request["image_path"]
            targets = request["targets"]

            image = Image.open(image_path).convert("RGB")

            result = worker.detect(
                image,
                targets,
            )

            response = {
                "ok": True,
                "result": result,
            }

        except Exception as exc:
            response = {
                "ok": False,
                "error": str(exc),
            }

        print(json.dumps(response, default=str))
        sys.stdout.flush()


if __name__ == "__main__":
    main()