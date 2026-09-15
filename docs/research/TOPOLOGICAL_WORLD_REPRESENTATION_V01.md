# Topological World Representation v0.1

## 1. Purpose

Step 25 is an engineering relational-topology sandbox, not evidence that AISTHESIS has topological cognition.

This standalone module compares exactly two explicit Step 24 snapshots. It does not integrate into production execution. New contracts are `TopologicalNodeState`, `TopologicalEdgeState`, `ComponentCorrespondence`, `TopologicalTransition`, and `TopologicalWorldState`; the entry point is `TopologicalWorldBuilder.build(previous_state, current_state)`. Every output has schema `topological-world-v0.1`.

## 2. Why relational topology may matter for partially observable world models

Relations can provide a compact description of how represented objects are grouped and how that description changes. Under partial observation, changes can reflect observation coverage, uncertainty, or representation choices. This implementation does not distinguish these explanations or establish predictive value.

## 3. Inputs

Both inputs must be actual `ComplexSystemsState` records with schema `complex-systems-v0.1`. There are no optional sources, hidden inputs, history queries, model calls, or memory writes. Exact object IDs are the only object identity mechanism.

## 4. Snapshot projection

Candidate statuses `present_candidate`, `possible`, and `indeterminate` are retained as current graph records. `unsupported` and `unavailable` records are excluded from the current graph but retained in the transition as indeterminate diagnostics, preventing false appearance/disappearance claims. The transition contains the union of source semantic keys; world arrays contain the current projection. Disappeared records therefore remain accessible through the transition.

No edges are added from proximity, class similarity, co-occurrence, a subsystem alone, shared stability assessments, or shared future branches. Source subsystem membership must be supported by explicit interactions.

## 5. Node semantics

Current nodes are exactly the explicit object IDs in current candidate interactions. A valid subsystem's exact object union is already covered by its referenced interactions. Isolated objects in the physical source are not projected. Transition node diagnostics also retain IDs in excluded source interactions. Presence is `represented`, `not_represented`, or `indeterminate`; it describes relational representation, not existence. A node's presence is indeterminate if any incident source relation has an indeterminate, unavailable, or unsupported status. No class, geometry, embedding, or tracker repair matches objects.

Missing relational structure is not evidence of physical absence.

## 6. Edge semantics

A topological edge is a representation of an explicit Step 24 candidate relation, not a verified physical or causal link.

The semantic key is `(tuple(sorted(object_ids)), interaction_type)`. The comparison preserves previous/current source interaction IDs, epistemic statuses, uncertainties, and complete source edge records in immutable provenance. Snapshot-specific source IDs need not agree. Compatible duplicate semantic records aggregate all source IDs. Different statuses for one semantic key are rejected as conflicting duplicates; although Step 24 can represent these in parallel, v0.1 deliberately declines to collapse their conflicting epistemic meanings.

## 7. Multi-edge/hyperedge semantics

Different interaction types over the same objects remain separate. An interaction involving more than two objects remains one hyperedge-like record; it is never expanded into pairwise edges. Undirected correspondence makes no direction claim.

## 8. Temporal correspondence

Nodes correspond by exact object ID, edges by semantic key, and components by object-set overlap. No additional identity inference is performed. Topological transitions retain previous/current complex state IDs, timestamps, session/frame, all diagnostic records, and uncertainty lineage.

## 9. Persistence semantics

Persistence means representation across the two supplied snapshots, not truth, confidence, permanence, or persistent homology.

A semantic key with represented candidate relations on both sides is `persistent`, including a change between `possible` and `present_candidate`. If either side is epistemically indeterminate, the comparison status is `indeterminate` even when both source ID lists are nonempty. The retained source lists still expose presence of the key in both snapshots.

## 10. Appearance/disappearance semantics

Edge appearance and disappearance describe representational change, not event occurrence.

A represented key only in the current snapshot is `appeared`; one only in the previous snapshot is `disappeared`. An explicit indeterminate, unavailable, or unsupported relation on either side makes the comparison `indeterminate`, including comparison against an empty snapshot. No global coverage/completeness metadata is invented: an empty supplied interaction list means an empty supplied representation, not a verified absence in the world.

## 11. Component correspondence

Construct a bipartite graph with previous and current Step 24 subsystem candidates as vertices. Connect opposite-side vertices iff their explicit object sets intersect. Deterministically traverse connected overlap groups, including isolated vertices. Each group produces one correspondence preserving all previous/current subsystem IDs, object union, source topological edge IDs, source component records, and uncertainty.

For a one-to-one group, equal sets are `persistent_component`; a strict current superset is `expanded_component`; a strict current subset is `contracted_component`. Equal membership does not imply equal internal relations. An isolated current component is `appeared_component`; an isolated previous component is `disappeared_component`.

## 12. Split/merge candidates

A group with multiple previous components and exactly one current component is `merge_candidate`. Exactly one previous and multiple current components is `split_candidate`. Competing overlaps therefore take precedence over expansion or contraction.

A component merge candidate is not object fusion.

A component split candidate is not physical separation.

## 13. Ambiguous reconfiguration

Many-to-many groups and one-to-one groups whose sets overlap without containment are `reconfigured_component` with `indeterminate` status. Any indeterminate source component, source edge, or overlapping uncertain diagnostic relation also makes correspondence status indeterminate. Otherwise correspondence status is merely `candidate`. The descriptive overlap category remains visible even when its status is indeterminate.

## 14. Uncertainty

Top-level uncertainty unions both source states, interactions, components, stability assessments, and transition candidates, plus generated edge/component uncertainty. Every node/edge/component receives source-state uncertainty and its directly relevant record uncertainty. Previous/current source-state uncertainty is retained separately in provenance; source edges and components preserve their own uncertainties.

Mandatory limitations are `representation_change_not_event_occurrence`, `topology_not_causality`, `two_snapshot_comparison_only`, and `missing_structure_not_physical_absence`. Indeterminate edges add `source_relation_indeterminate`; ambiguous component comparisons add `ambiguous_component_correspondence`. No uncertainty is converted into confidence or a score.

## 15. Alignment

Session and coordinate frame must match exactly. Timestamps must be finite, nonnegative, known, and strictly increasing; missing, equal, or reversed times are rejected. Scene IDs are snapshot labels, as in existing world-model fixtures (`scene1.0`, `scene2.0`), and may differ within the same session/frame. This is not a cross-scene identity matcher; callers must use session/frame to delimit a consistent comparison domain.

Child context fields, where present in Step 24, must match their owning snapshot. Source versions and content-derived identities are checked by reconstructing records for validation and comparing their content; malformed inputs are rejected rather than repaired. Source subsystem references must exist, their object set must exactly equal the union of their valid referenced edges, and those edges must be connected. Overlapping source components are rejected. These checks validate the supplied Step 24 contract, not the underlying physical truth or original sensor data.

## 16. Determinism

Frozen dataclasses, recursively frozen mappings, sorted string sets, sorted typed record collections, canonical JSON, and existing content-derived `stable_id` provide repeatable output IDs. Permuting source interaction/subsystem order or construction order leaves equivalent input/output serialization unchanged. No wall clock, randomness, external state, network, model, or dependency is involved. JSON exports are detached from immutable records.

## 17. Why topology is not causality

Graph connectivity is descriptive, not causal.

Topological distance is not physical distance.

No direction, propagation, intervention, or causal mechanism is inferred. This version does not calculate either kind of distance.

## 18. Why topology is not semantics

Topology does not provide semantics by itself.

Interaction types are copied from Step 24. Connectivity does not supply object meaning, goals, hidden identities, or physical explanations.

Step 25 does not infer hidden actors.

## 19. Why no scalar topology/complexity score

No global change, topology, complexity, node importance, or degree-based score is computed. Discrete descriptive categories remain auditable without implying a calibrated physical measurement.

## 20. Why no TDA/persistent homology in v0.1

Step 25 does not compute entropy, chaos, criticality, attractors, tipping points, Betti numbers, or persistent homology.

No filtration, Euler characteristic, homology, or topological data analysis machinery is introduced. Two-snapshot persistence is a representational label only.

## 21. Relationship to Step 24

Step 24 is the only input authority. Step 25 reads candidate interaction structure without rewriting it. Stability and transition assessments contribute uncertainty but never generate edges or topology-derived physics claims.

Step 25 does not select future branches.

Step 25 does not update Bayesian beliefs.

Step 25 does not modify ExperienceLearning.

## 22. Relationship to future Active Perception

Future Step 26 may evaluate structural change as one optional uncertainty-reduction cue. Step 25 requests no camera action, selects no attention, moves no sensor, emits no control command, and ranks no region of interest.

## 23. Relationship to Experiment 001

No better hidden-state inference, earlier anticipation, better TTA, topology benefit, hidden-actor anticipation, or scientific support is established. Later ablation may test whether relational temporal structure helps prediction. No Experiment 001 success is claimed.

## 24. Explicit non-goals

No hidden actor inference, learned matching, causal discovery, branch choice, Bayesian feedback, ExperienceLearning mutation, production orchestration, UI integration, persistent memory, new dependency, external topology repository, or model is added. Only the standalone module, dedicated tests, and this document are created. Existing files and old generated/untracked artifacts are untouched. Nothing is staged, committed, pushed, or tagged.

## 25. Validation status

Dedicated synthetic suite: **69 passed**. Coverage includes semantic persistence with changed source IDs, all eight component categories, many-to-many ambiguity, multi-edges, hyperedges, uncertainty and exclusion diagnostics, malformed inputs/references, temporal and child alignment, immutable provenance, detached serialization, input non-mutation, ordering determinism, and absence of feedback/scalar outputs.

Requested regressions: **969 passed** (38.30 seconds). Step 24: 55; Step 23: 87;
Step 22: 80; Step 21: 62; Step 20: 103; Step 19: 117; Step 18: 144;
Step 17: 93; world-model suites: 228 (physical world 31, physics constraints 45,
latent physical state 52, hybrid world state 39, common evidence state 61).
Together with the dedicated suite, **1,038 tests passed**.

Final `git diff --stat` is empty because all three additions are untracked.
`git status --short` shows those three additions alongside the pre-existing
untracked artifacts; no tracked production files changed and nothing was staged.

- IMPLEMENTED
- TESTED (synthetic contract tests)
- NOT YET TASK-LEVEL EVALUATED
- NOT YET EXPERIMENTALLY SUPPORTED
