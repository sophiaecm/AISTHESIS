from fifth_layer.perception.occlusion import extract_occlusion_evidence


detections = [
    {
        "class_name": "person",
        "box_xyxy": [100, 100, 300, 400],
    },
    {
        "class_name": "box",
        "box_xyxy": [220, 200, 360, 350],
    },
]

occlusion_evidence = extract_occlusion_evidence(
    detections=detections,
    image_width=640,
    image_height=480,
)

print("OCCLUSION EVIDENCE:")
print(occlusion_evidence)   