from fifth_layer.perception.smolvlm_scene import SmolVLMScenePerception

perception = SmolVLMScenePerception(
    model_id="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    device="cuda",
)

world_state = perception.perceive("test.jpg")

print(world_state)
print(world_state.data)