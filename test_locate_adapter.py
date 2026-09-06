from fifth_layer.perception.locate_anything import LocateAnythingPerception

locator = LocateAnythingPerception()

world_state = locator.perceive(
    "test.jpg",
    targets=["person", "plant"],
)

print(world_state)
print(world_state.data)