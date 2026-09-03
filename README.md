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

## Fifth Layer Engine v0.3

Fifth Layer Engine is model-independent. Its core architecture does not
require a language model. Reasoning strategies are interchangeable
modules.

Version 0.3 separates inference from orchestration by introducing a
reasoner interface, `BaseReasoner` (`fifth_layer/reasoners/base.py`).
`FifthLayerEngine` no longer contains any model-specific inference
logic — it simply calls into whatever reasoner it is given:

- `infer_expected_consequences(world_state)`
- `infer_latent_state(world_state, expected_consequences)`
- `infer_future_state(latent_state)`

`PlaceholderReasoner` (`fifth_layer/reasoners/placeholder.py`) implements
this interface with the same trivial logic used in v0.2, and exists only
to validate the architecture. It is the engine's default reasoner, but
any `BaseReasoner`-compatible object can be passed in instead:

```python
engine = FifthLayerEngine(reasoner=PlaceholderReasoner())
```

`PredictionError` comparison logic remains in `FifthLayerEngine` for now.

Future reasoning implementations may include:

- physical rules
- probabilistic inference
- learned predictive models
- sensor-derived models
- optional LLM adapters (e.g. for OpenAI, Gemini, Claude, DeepSeek)

None of these are implemented yet, and no such reasoner will ever be
required by the core engine — they are opt-in, pluggable modules only.

See `example_v03.py` for a runnable demonstration that explicitly
instantiates `PlaceholderReasoner`:

```bash
python3 example_v03.py
```

