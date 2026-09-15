# Active Perception Core v0.1

## 1. Purpose

An observation target candidate identifies a represented uncertainty that may benefit from additional observation; it does not identify a hidden physical fact.

`ActivePerceptionCore.build(topological_state, complex_systems_state=None, physical_state=None)` returns a frozen `ObservationRequestPlan` containing frozen `ObservationCue` and `ObservationTargetCandidate` records. Output schema is `active-perception-v0.1`. The module is standalone and is not connected to production orchestration.

## 2. Active perception in AISTHESIS

The only question answered is: given current represented uncertainty and structural ambiguity, what would be useful to observe again? The answer is an advisory symbolic candidate, with explicit source lineage and categorical routing priority. It does not decide where a camera should move, which actor exists, or what the true world state is.

## 3. Why reobservation can matter under partial observability

Additional observations may help resolve an explicit relation, grouping, or tracking uncertainty. This is a possible benefit, not measured improvement. Changes in representation can arise from coverage and uncertainty; the core does not resolve their physical explanation.

Missing evidence is not negative evidence.

## 4. Inputs

Required: actual current `TopologicalWorldState`, schema `topological-world-v0.1`.

Optional: current `ComplexSystemsState`, schema `complex-systems-v0.1`; current `LatentPhysicalState`, actual committed schema **`latent-physical-state-0.1`**. The brief's suggested spelling with `v` is not the frozen latent schema. Physical support requires complex support because Step 25 exposes no direct physical lineage. Topology alone suffices for topology cues.

No history retrieval, hidden input, external model, VLM description, learned embedding, user identity, or memory lookup is used.

## 5. Cue families

| Cue | Exact extraction rule |
|---|---|
| `indeterminate_relation` | Transition edge status is indeterminate or its current epistemic statuses include indeterminate; alternatively a supplied complex interaction is indeterminate. |
| `relation_change_ambiguity` | Transition edge status is appeared, disappeared, or indeterminate. |
| `component_reconfiguration` | Correspondence is reconfigured, split, or merge, or its status is indeterminate. |
| `subsystem_uncertainty` | Supplied complex subsystem status is indeterminate. |
| `stability_uncertainty` | Supplied stability assessment status is indeterminate; objects come from its explicitly referenced subsystem. |
| `transition_uncertainty` | Supplied transition status is indeterminate or pattern is unresolved_transition; objects come from its referenced subsystem. |
| `physical_state_uncertainty` | Supplied object/relation/dynamic has nonempty explicit record uncertainty, or a dynamic explicitly represents tracking_or_motion_discontinuity with observed/estimated/possible/indeterminate/unknown status. |

Global uncertainty is propagated but does not by itself create a target. Generic object class, proximity, motion, salience, missing objects, and hidden-actor labels are not cue rules. Generic inherited limitation strings on a determinate subsystem do not create a subsystem cue. Unsupported/unavailable discontinuity without explicit uncertainty is not a positive cue. Stability being unstable is not interpreted as danger.

The current transition is consumed directly, including previous-only disappeared edge diagnostics. Those IDs identify previously represented objects; they do not assert present existence or hidden actors. No temporal topology is recomputed.

## 6. Symbolic target semantics

Implemented scopes: `relation`, `subsystem`, `object`, `object_set`. Topological/complex edge cues use relation scope and preserve the interaction type. Component, subsystem, stability, and transition cues use subsystem scope. Physical cues use object scope for one explicit ID, otherwise object_set. No topological_region scope is invented without a grounding contract.

Targets contain only exact cue object IDs. Hyperedge object sets remain intact. Different relation types remain distinct even for the same object pair. No geometry, source bounding boxes, attribute values, masks, sensor poses, or trajectories are copied into output provenance.

## 7. Target aggregation

The deduplication key is `(target_scope, sorted object_ids, sorted relation_types)`. Only exact keys merge. Cue IDs and source IDs are unioned and sorted; repeated uncertainty strings cannot multiply targets. Cue identity is content-derived. Multiple layers repeating one relation merge lineage without being treated as independent evidence.

Latent mapping records have no native record ID. Their source references are explicitly labeled `symbolic_content_reference`, hashed from the latent state ID, collection, explicit object IDs, status, signal, uncertainty, and derived_from references. These are reference fingerprints, not invented upstream record IDs; geometry and class attributes are excluded.

## 8. Categorical priority

Reobservation priority is an advisory routing category, not probability, confidence, risk, danger, or truth.

| Band | Rule, evaluated in order |
|---|---|
| high | An indeterminate or reconfigured component cue explicitly references at least one indeterminate topological edge through source_edge_ids. |
| medium | At least one indeterminate relation, component, subsystem, stability, or unresolved transition cue. |
| low | Only representational change and/or bounded physical uncertainty cues. |
| indeterminate | Fallback if the supported cue set does not justify another category. |

The high rule is directly related structural corroboration; it does not assert statistical independence. Merely counting cue families, copied layers, or uncertainty strings never raises priority. In particular two cue labels derived from one indeterminate edge remain medium. A component's related edge may involve a subset of its objects, but that edge cue is not merged into the larger component target. No weighted sum, floating score, posterior, ExperienceLearning score, or learned ranking is used.

## 9. Empty-plan semantics

An empty observation plan does not imply that the world is certain.

It means only that no reobservation target is justified by these rules for the supplied inputs. A persistent determinate relation alone does not create a target. Optional-layer absence is recorded, not filled in.

## 10. Uncertainty

Every cue retains source record uncertainty and explicit epistemic limitations. Each target unions its cue uncertainty. The plan unions topological uncertainty, supplied complex/physical top-level uncertainty, all emitted cue uncertainty, and core limitations. Source top-level uncertainty is separately preserved by layer in plan provenance. Record uncertainties that create no cue are not converted into fabricated object-level observation needs.

Core limitations: target_not_truth; priority_not_probability_confidence_or_risk; uncertainty_not_hidden_actor; missing_evidence_not_negative_evidence; empty_plan_not_certainty; observation_request_not_control.

## 11. Provenance

Every cue preserves source_layer, source_record_ids, source_status, source_object_ids, source_uncertainty, and epistemic_limitation. Topological relation cues additionally retain previous/current epistemic statuses. Component cues retain correspondence_type and directly_related_indeterminate_edge_ids. Physical cues retain source_collection, source_signal, source_reference_kind, and discontinuity. Transition cues retain pattern_type.

Targets preserve cue/source IDs, rationale_code, epistemic_limitation, priority_policy, and expected_benefit_measured=False. Rationale codes are may_reduce_relation_uncertainty, may_reduce_component_ambiguity, and may_reduce_physical_state_uncertainty; none promises success.

Plan provenance states:

```text
advisory_only = True
executable_control = False
hidden_actor_inference = False
causal_discovery = False
Bayesian_feedback = False
experience_learning_feedback = False
branch_selection = False
information_gain_measured = False
external_grounding = False
validation = engineering_contract_only
```

It also records optional complex_support/physical_support availability and separate source_uncertainty by layer. The plan always references topological and complex source IDs; its physical source ID is known from complex support or None when that support is absent.

## 12. Alignment

Topology is the current authority. Validate schemas, source content-derived identities, transition/current session/frame/time/source-ID consistency, exposed current scene lineage, node/edge/component references, current projection completeness, and edge source provenance where exposed.

Optional complex support must exactly match current scene, timestamp, session, frame, and source complex ID. Its frozen Step 24 structures are validated with the existing read-only Step 25 validation helper. Current topological interaction references must match supplied complex interaction IDs, object sets, relation types, and statuses; subsystem references must exist.

Optional physical support requires complex support, exact scene/time/session/frame, and latent_state_id equal to complex source_physical_state_id. Physical relation/dynamic/constraint references must name explicit current objects; supplied complex interaction object IDs must exist in the physical state. If complex provenance exposes its original physical snapshot, supplied content must match after canonicalizing top-level record collection ordering and duplicates. The latent contract itself has a supplied ID rather than a constructor-recomputed hash; exposed snapshot comparison supplies an additional content check.

Previous and current scene labels may differ under Step 25 semantics, but optional support is for the current snapshot and must match its scene exactly. No mismatch is silently reconciled. Historical-only physical dynamic references are rejected in optional physical support because this version cannot safely ground them to current objects. Disappeared topology diagnostics remain supported without inventing physical identity.

## 13. Determinism

Frozen dataclasses, recursive mapping freezing, existing stable_id and bounded_plain helpers, sorted IDs/types/uncertainty, and canonical JSON provide deterministic detached serialization. Cues sort by cue_id. Targets sort high, medium, low, indeterminate, then target_id. Equivalent input collection permutations do not change output. Builds do not mutate sources. There is no UUID, random number, wall clock, unstable repr, network, external model, or global write.

## 14. Why target != truth

An appeared or disappeared topological relation describes representational change, not physical event onset or termination.

A split or merge candidate does not imply physical separation or fusion.

A target asks about represented uncertainty; it is not a verified physical or causal fact. Structural ambiguity is not a discovered causal ambiguity. Current uncertainty is not a future outcome. Instability is not danger.

## 15. Why uncertainty != hidden actor

Uncertainty does not imply a hidden actor.

Step 26A does not infer hidden actors.

No cue proposes looking behind an object because a person might be hidden. Explicitly ambiguous object sets can be reobserved without adding any unreferenced object.

## 16. Why active perception != control

Step 26A emits no sensor, camera, robot, gaze, or actuator command.

Observation need is not action need. The plan is advisory representation only and has no executable control behavior.

## 17. Why no numerical information gain in v0.1

Step 26A does not measure information gain or entropy reduction.

There is no calibrated observation model or empirical benefit estimate here. Categorical rationale codes describe a possible use of reobservation, not measured information gain.

## 18. Relationship to Step 25

Step 25 supplies current topology and temporal ambiguity. Step 26A reads it without changing its contracts, reconstructing temporal matching, or feeding scores back.

Step 26A does not select future branches.

Step 26A does not update Bayesian beliefs.

Step 26A does not modify ExperienceLearning.

## 19. Relationship to Step 26B LocateAnything grounding

Future Step 26B may map a symbolic candidate to a grounded observation target with a suitable adapter. LocateAnything is not integrated, downloaded, or called. Even later grounding would not itself be a motor command. This version emits no spatial grounding.

## 20. Relationship to Step 27 uncertainty/calibration

Broader uncertainty and calibration belong to future Step 27. This core neither estimates calibration nor becomes a posterior/risk/confidence engine.

## 21. Relationship to Experiment 001

No improved hidden-state inference, earlier anticipation, improved TTA, better prediction, hidden actor detection, or scientific benefit is established. Later experiments may test whether reobservation cues improve inference. No Experiment 001 success is claimed.

## 22. Explicit non-goals

No autonomous control, physical action, sensor motion, hidden actor inference, causal discovery, future branch selection, Bayesian update, ExperienceLearning mutation, prediction outcome evaluation, persistent memory write, new dependency/repository/model, network call, production integration, Mini-Lab edit, or paper-input edit. Exactly the standalone module, dedicated tests, and this document are created. No existing files are modified, and nothing is staged, committed, pushed, or tagged.

## 23. Validation status

Dedicated tests: **74 passed**. Synthetic coverage includes immutable contracts, schema, deterministic IDs/JSON and ordering, detached serialization, non-mutation, alignment/lineage failures, cue families, exact scope aggregation, empty plans, categorical priorities, hyperedges/multiedges, missing source references, absent geometry, and disabled feedback/control policy.

Requested regressions: **1,038 passed** (20.07 seconds). Step 25: 69; Step 24: 55;
Step 23: 87; Step 22: 80; Step 21: 62; Step 20: 103; Step 19: 117;
Step 18: 144; Step 17: 93; world-model suites: 228 (physical world 31,
physics constraints 45, latent physical state 52, hybrid world state 39,
common evidence state 61). Including the dedicated suite, **1,112 tests passed**.

Final `git diff --stat` is empty because the three additions are untracked.
`git status --short` shows exactly these three new files beyond the pre-existing
untracked artifacts. No tracked production files changed and nothing was staged.

Step 26A is an engineering active-perception representation layer, not scientific evidence that active perception improves AISTHESIS.

- IMPLEMENTED
- TESTED (engineering contracts)
- NOT YET TASK-LEVEL EVALUATED
- NOT YET EXPERIMENTALLY SUPPORTED
