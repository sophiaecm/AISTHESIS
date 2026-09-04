"""Example for Fifth Layer Engine v0.9 using VisualPerception."""

from fifth_layer.perception.visual import VisualPerception


perception = VisualPerception()

world_state = perception.perceive("test_image.jpg")

print(world_state)