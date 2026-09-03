"""Runnable example for Fifth Layer Engine v0.3.

Demonstrates that FifthLayerEngine is model-independent: it is driven
by an explicitly-provided reasoner (PlaceholderReasoner here) rather
than any built-in, model-specific logic.

Pipeline:

    WorldState -> ExpectedConsequences -> LatentState -> FutureState

and then compares a later, actually observed WorldState against the
predicted FutureState to produce a PredictionError.

Run with:
    python3 example_v03.py
"""

from fifth_layer import FifthLayerEngine, PlaceholderReasoner, WorldState

# The reasoner is chosen explicitly and passed into the engine. Any other
# BaseReasoner implementation could be used here instead, with no changes
# to FifthLayerEngine.
reasoner = PlaceholderReasoner()
engine = FifthLayerEngine(reasoner=reasoner)

# Step 1: an initial observed WorldState.
initial_state = WorldState(timestamp=0.0, data={"temperature": 20, "light": "on"})

result = engine.step(initial_state)

print("Expected Consequences:", result["expected_consequences"])
print("Latent State:", result["latent_state"])
print("Future State:", result["future_state"])

# Step 2: a later WorldState representing what was actually observed.
observed_state = WorldState(timestamp=1.0, data={"temperature": 22, "light": "on"})

prediction_error = engine.compare(observed_state)

print("Prediction Error:", prediction_error)
