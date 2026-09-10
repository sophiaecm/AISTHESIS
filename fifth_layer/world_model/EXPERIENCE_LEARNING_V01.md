# Experience Learning v0.1

Experience Learning v0.1 performs deterministic retrieval-based use of
historical ExperienceEpisodes. It does not train neural models, update
weights, calibrate probabilities, or treat historical experience as
ground truth.

Implemented locally against `f8f76ae`. No commit, push or tag was performed.

## Architecture and scope

Current SceneState + EvidenceBundle + existing HypothesisSet -> ExperienceQuery
-> read-only ExperienceMemory snapshot -> ExperienceRetriever ->
ExperienceEvidenceProvider -> separate bounded ranking context.

The frozen prediction/outcome evaluator, sensory semantics, ExperienceEpisode
schema, and existing hypothesis score rules are unchanged. There are no new
models, embeddings, training, replay training, calibration, active perception,
task heads, neural updates or second memory implementation.

Inspection covered `world_model/` including memory, prediction records,
prediction/outcome comparison, evidence/providers, hypotheses/generator,
trajectories, scene adapter and sensory integration; `evaluation/experience.py`
and existing world-model/sensory/experience/harness tests. Existing records have
issuance-time `hypothesis_provenance.score_inputs`; these are reused for matching.
Historical outcome geometry/descriptions are not used as similarity features.

## Retrieval, similarity and leakage

`ExperienceQuery.from_current(scene, evidence, hypothesis)` extracts only
explicitly associated structured features available at issuance. Supported
features are hypothesis type, object class, motion state/direction, explicit
visibility, measured overlap flag, and modality-qualified sensory status and
consequence. Unknown/conflicting source values remain unavailable. It does not
parse text, infer missing classes, guess identities or create actor/event types.

`ExperienceRetriever.retrieve(query, memory, current_time=...)` requires exactly
the query issuance time. It uses a locked, non-mutating
`ExperienceMemory.snapshot(at_time=...)`; this read never calls the memory clock,
cleans storage, refreshes TTL, reorders records or writes inference results.
Admission and query times must share a clock domain. Use an explicitly injected
scene/replay clock when populating memory; do not mix scene timestamps with the
default monotonic admission clock. Existing memory operations keep their original
512-entry maximum, 60-second TTL, duplicate protection, eviction and locking.

Eligibility is fail-closed. PredictionRecord, OutcomeRecord and
PredictionEvaluation summaries must validate and agree with episode links,
statuses and timestamps. Source, target, observation and evaluation timestamps
must all be known and **strictly earlier** than current_time. Equal-time results
are excluded. Observation must follow issuance and be known by evaluation time.
Explicit future event times and event times later than their evaluation are
rejected. Current scene/hypothesis records, pending, expired, unevaluable and
incomplete legacy records are excluded. No target observation or current
evaluation is read to construct a query.

Similarity is `matched_count / comparable_count`. Missing fields, `unknown`, and
`unavailable` are excluded from both numerator and denominator. Matching only a
hypothesis name is insufficient: at least one structural feature must also
match. Default minimum similarity is 0.5; default top_k is 3, configurable from
1 to 8. Ordering is descending similarity then canonical episode ID. Each result
reports matched, conflicting and unavailable feature names, both raw component
values, status, similarity, historical times, evidence IDs and policy version.
The score is a heuristic, never a probability.

Unobservable, insufficient and partially supported completed records can supply
context but never a positive/negative contribution. Records with unknown
completion times are not retrieved at all. All status names retain their frozen
meaning; historical contradiction does not establish physical impossibility.

## Evidence and bounded influence

The existing EvidenceItem/EvidenceBundle is reused with new
`EvidenceSource.EXPERIENCE='experience'`. Items preserve episode ID, historical
prediction type/status/metrics, component matches, heuristic similarity,
historical times and retrieval provenance. `current_observation=False` is
explicit; confidence is unknown and `supports`/`contradicts` are empty. Historical
sensory `expected`/`inferred` labels remain in provenance, never current observed
footsteps/contact/heat. Unavailable modalities are never negative evidence.

`experience_context(..., current_time=..., enabled=False)` is the opt-in entry
point. Disabled mode never reads memory. Both modes retain the exact original
HypothesisSet, including confidence, prior/posterior scores and alternatives.
Only a separate ranking view exposes original score, contribution and final
heuristic score; there is no probability normalization or winner selection.
The generator ignores experience items even if a caller merges them into an
EvidenceBundle, preventing accidental candidate creation or score changes.

For an active existing hypothesis, contribution additionally requires current
associated, contemporaneous physical support and no current opposing evidence.
Explicit predicted-track records cannot provide that physical support. Missing
support or any current contradiction forces contribution to zero. Histories
with conflicting structural components remain visible but have zero influence.

Historical status is not a reward. A supported/contradicted status contributes
only when the relevant comparison metric agrees: motion_state_match for motion
or stop, visibility_match for occlusion or reappearance, otherwise event_match.
A position-only contradiction cannot penalize a motion/event hypothesis.
Eligible signed similarities are averaged, not summed across episodes:

```text
bound = min(configured_max, 0.05 * original_heuristic_score)
contribution = clamp(bound * mean(signed_similarities), -bound, bound)
final_heuristic_score = original_heuristic_score + contribution
```

Configured maximum cannot exceed 0.05. Repetition cannot accumulate unbounded
support. Competing supported/contradicted experiences can coexist and cancel;
neither deletes a candidate. A ranking value such as 1.05 is explicitly an
unnormalized heuristic, not a probability. No original Hypothesis is mutated.
Historical ball-to-human experiences cannot create hidden humans, collisions,
falls, children, contact, sound sources or unseen objects.

## Offline A/B comparison and reproduction

`evaluation.experience_learning` accepts saved sensory results and an existing
experience report. It retains current input bytes/fields, hypothesis set and
physical evidence for both `NO_EXPERIENCE` and `EXPERIENCE_ENABLED`. No detector,
VLM, reasoner, evaluator or hypothesis generator is rerun. Output includes
retrieved IDs/scores/features, experience evidence, affected hypotheses, original
and final heuristic scores, physical-support flags, provenance and leakage audit.

Only same-run, same-mode, same-source-video sessions are comparable. Different
run IDs are rejected because local video timestamps are not a shared chronology.
Episodes enter replay memory only after all required historical times have
passed; preloading future records cannot evict usable history. Capacity/TTL are
applied through existing memory operations and the read-only snapshot. Frame
selection is unchanged. Input/episode ordering does not change output.

Manual commands, after review, using matching combined-run files:

```powershell
.venv/Scripts/python.exe -m evaluation.experience_learning results_case_a_combined_sensory.json --history results_case_a_experience.json --output results_case_a_experience_learning.json
.venv/Scripts/python.exe -m evaluation.experience_learning results_case_b_combined_sensory.json --history results_case_b_experience.json --output results_case_b_experience_learning.json
```

For Fifth Layer-only runs, first obtain a matching history with the unchanged
`evaluation.experience` utility and a new filename, then use that history:

```powershell
.venv/Scripts/python.exe -m evaluation.experience results_case_a_fifth_layer_sensory.json --output results_case_a_fifth_layer_experience.json
.venv/Scripts/python.exe -m evaluation.experience_learning results_case_a_fifth_layer_sensory.json --history results_case_a_fifth_layer_experience.json --output results_case_a_fifth_layer_experience_learning.json
```

Output requires a new `*_experience_learning.json` path and exclusive creation.
Existing inputs/results are never overwritten. CLI reports both source-file
SHA-256 hashes. No real Case A/B ablation or video/VLM evaluation was run during
implementation; validation used deterministic synthetic records.

Limitations: exact structured matching has no semantic generalization; incomplete
historical source context is not reconstructed. Sparse histories may yield no
retrieval or zero influence. Source timestamps/record provenance must be truthful;
the system cannot independently verify logging latency. Cross-run transfer needs
an explicit shared chronology and is not provided. No ground-truth accuracy or
improvement claim follows from these ablations.

## Exact files and validation

Modified:

- `fifth_layer/world_model/evidence.py`: one experience source enum member.
- `fifth_layer/world_model/experience_memory.py`: read-only explicit-time snapshot.
- `fifth_layer/world_model/hypothesis_generator.py`: ignore experience evidence
  during candidate generation; existing scoring/sensory rules unchanged.

Created:

- `fifth_layer/world_model/experience_learning.py`
- `evaluation/experience_learning.py`
- `test_experience_learning.py`
- `evaluation/tests/test_experience_learning.py`
- `fifth_layer/world_model/EXPERIENCE_LEARNING_V01.md`

**430 tests passed: 377 existing + 53 new; 0 failures, 0 unittest skips.**
New tests: 41 retrieval/evidence/context tests and 12 offline ablation tests.
Existing suites include 66 World Model v0.1, 46 v0.2, 34 sensory, 52 prediction/
experience, 24 harness, 5 sensory postprocessing, 14 experience postprocessing,
and 136 other regression tests. Tests cover future/equal-time leakage, input
permutations, byte determinism, TTL/capacity, no clock calls/mutations, missing
features, epistemic status, competing histories, inference isolation, metric
relevance, bounds, direct contradictions and overwrite protection.

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$testModules = Get-ChildItem -Filter 'test_*.py' | Where-Object { (Get-Content -LiteralPath $_.FullName -Raw) -match 'unittest|def test_' } | ForEach-Object { $_.BaseName }
.venv/Scripts/python.exe -m unittest @testModules evaluation.tests.test_harness evaluation.tests.test_sensory evaluation.tests.test_experience evaluation.tests.test_experience_learning -q
```

Eight deterministic smoke scripts also passed: active_perception, auditory,
auditory_negative, occlusion_reasoner, orchestrator, semantic_conflict, temporal,
yolo_occlusion (their root `test_*.py` scripts). Five neural/service smoke scripts
were intentionally not run: `test_active_perception_loop.py`,
`test_locate_adapter.py`, `test_perception_fusion.py`, `test_smolvlm_core.py`,
`test_yolo_detector.py`. They are excluded from the isolated test count.

`git diff --check` passed. Production behavior with experience disabled is
unchanged. `live_app.py`, SmolVLM, detector/tracking, all reasoners/orchestrator,
sensory code, prediction/outcome evaluator, calibration, Episode schema and all
existing tests were untouched. No models/repositories downloaded, no training,
no commits/pushes/tags. HEAD remains `f8f76ae`.

All 12 existing Case A/B JSON files have identical pre/post SHA-256 hashes:

| File | SHA-256 |
| --- | --- |
| results_case_a_combined.json | A2C478A99C107C95EC07975517F8DED58963C8CDDBEE0052166B01C29141DE7D |
| results_case_a_combined_sensory.json | 12DF861878B6EE941E60360B4A89042065D564B20085AD96520ABC876CE2876B |
| results_case_a_experience.json | 3B979CAC0B563DEFA069FCF16C39BAD8D9FAABE87816BFF913B1E0600DF8C2F4 |
| results_case_a_fifth_layer.json | 1D0866C30DC4E98B2CC17DB25EE40D7829C4354058D7D87886B693D1C15D0759 |
| results_case_a_fifth_layer_sensory.json | F017D2FAAC4EB4EC7D59A9777B3F7E3A2FD79729B2B3FDBE92193B97598319C1 |
| results_case_a_vlm.json | 8C137B8E1F61C1F855F4FA0B33CFE2670620C35C323C60CA89DDDDB62EDA3F9E |
| results_case_b_combined.json | 955F4886D11987B08D06742038676C3B70705A256DF7C17ADABCAC62E6267B06 |
| results_case_b_combined_sensory.json | 0BB3B84AD83BF3A232477C185C455C4113062448668A5AF495087A78A33DF3D5 |
| results_case_b_experience.json | 0DD4D83B3F29844EE5BFF18F47BE002A305AA5339B673D46DEE337DB057EAEBC |
| results_case_b_fifth_layer.json | EA8FFC93F80C1EEA4C18FC78A0EE6BFF2AFE39DC6B4FB78B68AE552D14891C0E |
| results_case_b_fifth_layer_sensory.json | B0EF065DBCD14CF232A528725801BB87946CB7B01DBB1AC43A2BDCA0E21DBC54 |
| results_case_b_vlm.json | EE8882BF018788613E185F902D451EFCBA5A4AB704AF2E0133259EFC3297E000 |

The result JSONs, `test_videos/`, and `test_prediction_experience.py` were already
untracked at task start and were not created or modified by this task.
