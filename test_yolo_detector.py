from fifth_layer.perception.yolo_detector import YoloDetectorPerception

detector = YoloDetectorPerception(
    model_path="yolo11n.pt",
    confidence_threshold=0.20,
    device=0,
)

world_state = detector.perceive("test.jpg")

print(world_state)
print(world_state.data)