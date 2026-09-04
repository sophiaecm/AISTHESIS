"""Example for Fifth Layer Engine v0.19 temporal motion evidence."""

from fifth_layer.perception.temporal import extract_motion_evidence


previous_detections = [
    {
        "class_name": "person",
        "confidence": 0.90,
        "box": [100, 300, 200, 500],
    },
    {
        "class_name": "motorcycle",
        "confidence": 0.85,
        "box": [650, 350, 250, 500],
    },
]

current_detections = [
    {
        "class_name": "person",
        "confidence": 0.91,
        "box": [140, 300, 200, 500],
    },
    {
        "class_name": "motorcycle",
        "confidence": 0.86,
        "box": [620, 350, 250, 500],
    },
]

motion_evidence = extract_motion_evidence(
    previous_detections=previous_detections,
    current_detections=current_detections,
    image_width=1000,
    image_height=1573,
)

print("MOTION EVIDENCE")
print()

for item in motion_evidence:
    print(item)