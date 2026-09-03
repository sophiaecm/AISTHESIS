"""Example for Fifth Layer Engine v0.8 using ImagePerception."""

from fifth_layer.perception.image import ImagePerception


perception = ImagePerception()

world_state = perception.perceive("test_image.jpg")

print(world_state)