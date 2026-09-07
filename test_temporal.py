from fifth_layer.perception.temporal import extract_motion_evidence


previous_detections = [
    {
        "class_name": "person",
        "confidence": 0.90,
        "box_xyxy": [
            100,
            100,
            300,
            400,
        ],
    }
]


current_detections = [
    {
        "class_name": "person",
        "confidence": 0.91,
        "box_xyxy": [
            140,
            100,
            340,
            400,
        ],
    }
]


motion_evidence = extract_motion_evidence(
    previous_detections=previous_detections,
    current_detections=current_detections,
    image_width=640,
    image_height=480,
    delta_time=0.5,
)


print("TEMPORAL MOTION EVIDENCE")

for item in motion_evidence:
    print(item)