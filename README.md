# AISTHESIS
Predictive sensory inference for Physical AI.

## Fifth Layer Engine

The `fifth_layer` package provides the initial scaffold for AISTHESIS's
predictive inference core, built around five concepts:

- **WorldState** – the observed state of the world at a point in time.
- **ExpectedConsequences** – predicted effects/outcomes expected from a state.
- **LatentState** – an internal/hidden representation derived from world state.
- **FutureState** – a projected future state.
- **PredictionError** – the discrepancy between expectation and actual outcome.

This is a minimal, dependency-free scaffold — no inference logic or LLM
adapters are implemented yet.

## Fifth Layer Engine v0.2

Version 0.2 adds a minimal working pipeline, `FifthLayerEngine`
(`fifth_layer/engine.py`), that connects the five core concepts:

1. Accepts a `WorldState` as input.
2. Generates `ExpectedConsequences` using placeholder logic.
3. Builds a `LatentState` from the `WorldState` and `ExpectedConsequences`.
4. Projects a `FutureState` from the `LatentState`.
5. Lets a later observed `WorldState` be compared against that
   `FutureState` to produce a `PredictionError`.

All transformation logic is clearly marked as placeholder logic — there
is no machine learning or real sensory inference yet. See
`example_v02.py` in the repository root for a runnable demonstration:

```bash
python3 example_v02.py
```

