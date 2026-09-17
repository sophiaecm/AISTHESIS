# Experiment 001 Protocol v0.1

Step 28A: pre-experiment protocol for review and subsequent freeze.
Production reference: `e1c3cf0c80006334408d2f0180f3cd3e332a4ff6`, annotated tag
`evaluation-metrics-v0.1` (peeled commit verified). No experiment was run here.

```text
PREDICTION != OBSERVATION
POSSIBILITY != FACT
INFERRED ACOUSTIC EVENT != OBSERVED SOUND
EXPECTED SOUND != OBSERVED SOUND
INFERRED ACOUSTIC EVENT != PHYSICAL FACT
CROSS-MODAL CONSEQUENCE != SENSOR MEASUREMENT
GROUND TRUTH != MODEL OUTPUT
MISSING ANNOTATION != NEGATIVE EVENT
UNEVALUABLE != INCORRECT
NO ANTICIPATION != TTA ZERO
EARLY PREDICTION != CORRECT PREDICTION
TTA ALONE != GOOD PERFORMANCE
SHUFFLED CONTROL != BASELINE TRUTH
METRIC RESULT != SCIENTIFIC VALIDATION
CONFIDENCE != PROBABILITY
[0,1] VALUE != PROBABILITY
```

## 1. Repository inspection: implementation versus intention

| Inspected source | Existing implementation | Boundary for this experiment |
| --- | --- | --- |
| `fifth_layer/prediction_evaluation.py`, calibration and their tests | Same-track geometric outcomes; correct/partial/incorrect, unevaluable and expired; finalization grace 0.25 seconds; explicit 1/0.5/0 calibration weights | Not a hidden-actor event annotation contract |
| Steps 27A, 27B and 27C modules, documentation and tests | Immutable uncertainty, issuance/calibration evidence and descriptive metrics | Confidence is not forecast probability; Step 27C TTA remains `protocol_required` |
| `physical_state_builder.py`, `physics_constraint_engine.py`, `latent_physical_state_builder.py` and corresponding contracts/tests | Image-plane physical attributes, deterministic constraints and latent summaries | No verified hidden-object truth or physical simulator |
| `cross_modal_consequences.py` and tests | Conditional candidates with modality, event family, status, source fields, uncertainty and `confidence=None`; acoustic `impact` from collision-risk geometry | No audio measurement or hidden-actor inference; vocabulary is limited |
| `multiple_futures.py` and tests | Conditional branches; internally builds an audited consequence inventory even when no external consequence bundle is supplied | Omitting one argument is not evidence of a clean acoustic ablation |
| `bayesian_belief_state.py` and tests | Branch-label distributions with caller-supplied priors/likelihood evidence | Not calibrated event forecasts; no automatic hidden-event predictor |
| `belief_prediction_bridge.py` and tests | Explicit branch selection and supported projection into existing prediction records | Does not project arbitrary acoustic consequences into hidden-actor predictions; unavailable content stays unavailable |
| `experience_learning_v02.py`, connectome v0.2 and tests | Bounded historical structural context and routing heuristics | Neither generates hidden actors; disable experience retrieval/update for this primary protocol |
| `evaluation/README.md`, adapters, harness, claims and tests | Same-media offline VLM/Fifth Layer mode separation, isolated workers, frame manifest and exact text-atom comparisons | No Experiment 001 ground-truth scoring; claim count is not quality |
| Existing `results_case_*` artifacts | Earlier harness output, including an existing SmolVLM2-500M-Video-Instruct configuration | Development material, not held-out Experiment 001 evidence; do not rescore or select samples from its outcomes |
| `VJEPA2_SANDBOX_V01.md` | Documentation of external sandbox execution and timing limitations | No V-JEPA production integration in Core; no new integration in Step 28A |

Repository searches covered Experiment 001, EXP001, TTA, Time-to-Anticipation,
anticipation, hidden actor, occlusion, ground truth, event onset, baseline,
ablation, silent scene, acoustic, cross-modal and FlyBrain. Existing Experiment
001 references describe an intended hypothesis and future evaluation boundary.
No fixed task protocol or completed Experiment 001 was found. The prior sandbox
documentation explicitly warns about future contamination in whole-video
sampling. Those previously inspected clips are development-only.

**FLYBRAIN IMPLEMENTATION NOT PRESENT IN CORE.**

Step 28A adds a protocol manifest and pure validation functions, not the missing
inference adapter, condition runner, dataset, labels or experimental evidence.

## 2. Research question and primary hypothesis

“Can reconstructing expected but unavailable sensory consequences improve
latent-state and near-future event inference under partial observability beyond
systems reasoning from the same visible evidence without that reconstruction?”

The controlled operational test asks whether inferred acoustic consequences of
visible physical dynamics help anticipate a subsequent actor/object appearance
or reappearance in a predefined region. An actor need not be human. This first
task measures near-future event anticipation; it does not directly measure the
correctness of all internal latent states.

The primary hypothesis is that correctly paired acoustic representation improves
valid anticipation relative to the otherwise identical acoustic-removal
condition, with false anticipation reported alongside coverage. Correctly paired
representations should also outperform the deterministic shuffle control if
sample-specific acoustic content contributes. This is a hypothesis, not a promise
or a claim of additional observed information: inferred representations are
functions of the same visual evidence.

## 3. Falsification and analysis commitments

Primary contrasts are `C_FULL - C_NO_ACOUSTIC` and
`C_FULL - C_SHUFFLED_ACOUSTIC`. Compare anticipation coverage and negative-trial
false-alarm fraction jointly. Unchanged or worse coverage, no advantage over
shuffle, or an apparent gain accompanied by indiscriminate false commitments
does not support the proposed contribution. Removal/shuffle invariance weakens
the acoustic-contribution interpretation. A versus B and A/B versus C describe
broader architecture differences and cannot isolate the acoustic contribution.

Report all contrasts and all exclusions, including negative results. No
post-hoc best condition, best subset, best horizon or score weighting is allowed.
TTA distributions alone cannot establish improvement, especially if different
conditions succeed on different trials. Report paired TTA differences on the
intersection of successfully anticipated positive trials, together with that
intersection size and each condition's overall coverage. This conditional
comparison must not stand in for performance on all positives.

This v0.1 protocol fixes descriptive estimands. Sample size, allocation and any
inferential statistics are not yet justified. Before test inference, a separate
dated/versioned dataset/run preregistration must set sample count, stopping rule,
split manifest and either an explicitly descriptive analysis or an inferential
plan with power assumptions, uncertainty estimates and multiplicity treatment.
No significance, equivalence or population-wide support may be claimed from this
protocol alone. No sample-size extension based on favorable/unfavorable outcomes.

## 4. Experimental unit and temporal record

One trial is one immutable silent visual sequence with one focal appearance
category and one predefined image-region query in a fixed monitoring interval.
The acquisition plan assigns the category/region task and interval before model
outputs are inspected. A full recording may contain initial visible exposure,
occlusion/partial visibility, and a later appearance. The monitoring interval
starts at a fixed acquisition-design frame, not a model-dependent time or a
ground-truth-onset-relative crop. Initial exposure may precede monitoring and is
available in the same prefix to every condition. Multiple independent focal
events require separate, preregistered trials with grouping by original recording
to prevent train/test leakage and false independence.

Required fields are sequence ID, split, media hash/reference, ordered frame
indices and source timestamps, fixed monitoring start, task category and region,
annotation status/provenance, per-frame visibility, optional prior observed
frame, optional inclusive occlusion interval, and positive onset frame.
Frame indices are zero-based positions in the immutable selected-frame manifest.
Actor annotation identity is retained in the external annotation, not supplied
as a hidden-object identity to inference. Primary matching uses **category and
region**, never guessed track-ID equivalence.

At every scheduled cutoff the inference input is the same sequence prefix from
frame zero through that cutoff, inclusive. No centered windows, future tokens,
full-clip summaries or ground-truth relative sampling. Every frame at or after
monitoring start with a complete subsequent 2-second observation window is a
scheduled decision. Scheduled cutoffs are determined by media duration/timestamps,
not event onset, predictions or condition. Reset state between trials and
conditions. Shared acquisition blocks use identical schedules and timestamp
grids; missing required observations invalidate the paired input comparison.

## 5. Partial observability and ground truth

Partial observability means the focal actor/event cannot be fully resolved from
the current admissible visual prefix: the target is outside the visible region,
behind an occluder, or only partially visible. Geometric tracker loss alone is
not annotated physical occlusion. Visible dynamics of other objects may support
conditional acoustic consequences without identifying a hidden actor.

Manual annotation must be completed independently of model output. Record media
hash, annotation version, annotator IDs, per-frame masks/visibility/category,
region geometry, prior observation/occlusion intervals, onset, complete follow-up,
ambiguity reason and adjudication. Two annotators independently label test trials
while blinded to condition outputs; disagreements are adjudicated before scoring
and unresolved cases remain ambiguous. No labels are fabricated in Step 28A.

Distinguish `OBSERVED BEFORE OCCLUSION`, `OCCLUDED / PARTIALLY OBSERVABLE`,
`REAPPEARANCE / TARGET EVENT`, `NO TARGET EVENT`, and `AMBIGUOUS / UNEVALUABLE`.
Code uses visibility labels `absent`, `occluded`, `partial`, `visible`,
`ambiguous`, `unannotated` and trial statuses `positive`, `negative`, `ambiguous`,
`missing_annotation`, `censored`. A negative has complete annotation of no focal
appearance during monitoring and follow-up. It need not prove the actor does not
exist behind the occluder. A recording that ends before the planned follow-up is
censored, not negative. Missing annotation is never evidence of no event.

Ground-truth records are evaluator-only. Shared public task instructions and
the global category/region vocabulary are allowed; per-trial hidden identities,
onset, positive/negative designation and reference masks are never inference input.

## 6. Objective event onset

Task onset is the **first frame in the monitoring interval** with at least
16 connected manually annotated visible target pixels inside the preregistered
region, confirmed by another qualifying frame immediately afterward. The onset
timestamp is that first frame's source timestamp, not the confirmation frame.
This is a prospectively chosen visible-appearance boundary, not acoustic onset,
underlying physical event start, or the model's detection time.

`visible` in the annotation means this pixel criterion is met and the category
is resolved by independent annotation; `partial` denotes weaker visible support
below that criterion. A lone qualifying frame not confirmed next frame is
ambiguous rather than silently moved to a later convenient onset. Source
resolution is fixed within acquisition blocks; do not resize per condition to
alter visibility. The annotation audit must verify masks/pixel counts. The
lightweight code validates the resulting symbolic labels, first-frame rule and
confirmation, but does not load images or validate masks itself.

Use the same adjudicated onset for all conditions. A reappearance may follow an
initial observed interval and an annotated occlusion interval; first appearance
does not require pretending the target was previously observed. Event identity
remains the predefined focal category/region, with exact identity secondary only
if a future preregistered identity contract justifies it.

## 7. Valid anticipation and commitment policy

At each scheduled cutoff, return exactly one structured decision: `anticipate`
with one actor category and one region, or `abstain`. Event kind is fixed to
appearance. A free-text mention, list of alternatives, inferred sound alone or
unselected future branch is not a commitment. No confidence/probability threshold
is used. A future adapter must apply a fixed, preregistered top-one or abstention
policy before evaluation; never select a matching branch after seeing truth.

A valid anticipation requires the exact category/region match, known source
times, issuance equal to the observation cutoff timestamp, evidence through that
cutoff only, and `issuance < onset <= issuance + 2.0 seconds`. All conditions
use this same horizon and rule. Later processing wall-clock time is logged
separately. A commitment with an expired horizon cannot be revived by a later
event. A new commitment at a later cutoff is a new decision, not retroactive
editing of the old one.

Exact retransmission of the same prediction ID and identical content is counted
once. Rewritten IDs or two different decisions at one condition/cutoff are
protocol violations, rejected for repair/audit before scoring. Across cutoffs,
contradictory or wrong earlier commitments remain in false-anticipation counts;
later correction does not erase them. A complete log must include abstentions.
Missing log entries are failures of evaluability, not inferred abstentions.

## 8. Time-to-Anticipation

For a positive trial with complete eligible log:

`TTA_seconds = t_onset - min(t_issuance of valid anticipations)`.

The value is strictly positive and at most 2.0 seconds. Use source timestamps,
not frame counts or wall-clock processing latency. Frame numbers index evidence;
for validated constant-frame-rate sources the existing harness uses frame/FPS.
Variable-frame-rate sources require an independently validated presentation-time
manifest before use and are outside the current Stage 1 harness path.

| Situation | TTA |
| --- | --- |
| Positive with one or more valid commitments and complete log | Positive seconds from earliest valid issuance |
| Positive with no valid anticipation, including only at/after-onset predictions | `None`, `no_anticipation` |
| Complete negative trial | `None`, `negative_trial` |
| Ambiguous, missing annotation or censored trial | `None`, `annotation_unavailable` |
| Missing frame/issuance/evidence timestamp | `None`, `missing_timestamp` |
| Missing scheduled decisions | `None`, `incomplete_prediction_log` |
| No cutoff with full horizon follow-up | `None`, `no_eligible_windows` |

Invalid known chronology is rejected; it is never repaired with guessed times.
Individual late windows without full follow-up are `right_censored_window`, even
if a positive event is already known. At/after-onset predictions are not pre-event
anticipations and are excluded from pre-event commitment metrics. Their records
are retained. Excluded TTA never becomes zero, negative lead, or a synthetic
failure time. No survival-analysis interpretation or censoring correction is
claimed in v0.1.

## 9. Negative trials and false anticipation

For a resolved negative trial, every scheduled `anticipate` is false. For a
positive trial, pre-event commitments with wrong category/region or target onset
outside the issued horizon are false. This task scores the focal appearance,
not unrelated genuine events elsewhere. Ambiguous truth, censored windows and
missing logs must not become incorrect predictions.

Count each resolved commitment once per scheduled cutoff, including repeated
alarms at successive cutoffs. Also report how many negative trials have at least
one false commitment. This makes constant alarm behavior visible even if its
earliest matching statement obtains a large TTA. All conditions have equal
decision opportunities; no global composite trades false alarms against lead.

## 10. Conditions and fairness

| Condition | Allowed representation |
| --- | --- |
| A_VISUAL | Existing learned visual/VLM capability; common detector/tracking geometry may also be supplied; no explicit AISTHESIS physics constraints or experimental acoustic packet |
| B_VISUAL_PHYSICS | Same visual evidence plus frozen explicit physical representation/constraints; no experimental acoustic packet |
| C_FULL | Common visual evidence plus physical, latent/future/belief context and inferred acoustic representation |
| C_NO_ACOUSTIC | Same C downstream logic and budgets with acoustic representation and all derived acoustic inputs removed |
| C_SHUFFLED_ACOUSTIC | Same C logic with only acoustic payload replaced by an eligible donor-prefix payload |

The existing available Core learned baseline path is SmolVLM scene perception;
earlier harness artifacts record SmolVLM2-500M-Video-Instruct. Use that legitimate
existing capacity without deliberately reducing its input or decoding budget.
There is no evidence establishing a stronger Core learned baseline or an
empirical ranking among checkpoints. Exact local checkpoint hash, prompt,
prefix-to-input adapter and generation settings must be preregistered and shared
where applicable. A sequence-capable task-output adapter is not implemented by
the existing frame harness; it must not be claimed operational now. V-JEPA2.1
remains an external sandbox comparator requiring a separate preregistered run,
not a silent replacement or integration into Core.

All conditions receive identical admissible visual bytes, cutoff, frame grid,
task vocabulary, horizon, target annotation, exclusions and matching policy.
Use fixed common seeds, independent fresh state and the same model/budget for
matched C contrasts. No online experience or calibration feedback from test
outcomes. Disable experience-learning retrieval/update in all primary conditions;
keep any prior/likelihood choices fixed from development data and disclose them.
No active perception acquires additional frames. Inference workers cannot access
annotation/evaluator files; evaluator runs only after immutable outputs are saved.
Audit input hashes, source timestamps, nested references and process isolation.
Latency comparisons, if later added, require identical hardware and separately
reported wall-clock processing latency and algorithmic anticipation time.

## 11. Acoustic payload and ablation boundary

Use only frozen, visually derived conditional acoustic candidates. Current v0.2
supports `impact` under collision-risk prerequisites; unsupported expectations
remain unavailable. Do not synthesize observed audio, novel actor identities or
new hidden-event capabilities. Primary experimental packet content is the sorted
multiset of `event_family`, `status`, and `uncertainty` for acoustic candidates.
Descriptions, object/scene IDs, physical/constraint reference IDs, numeric
confidence and provenance text are excluded from the downstream experimental
packet to avoid direct visual/identity cues or source mismatch acting as the
treatment. Candidate multiplicity and unavailable/empty status remain visible.
Complete original candidates and references stay in the evaluator's audit sidecar.

Other non-acoustic candidate channels are withheld uniformly from the experimental
representation input in all C variants. Common physical/latent state remains
available. Thus the matched contrast concerns a controlled acoustic representation,
not a larger bundle containing other changed modalities. Inferred acoustic content
is never described as recorded sound or ground truth.

Existing Multiple Futures and connectome records enforce source alignment, and
Multiple Futures internally regenerates consequences. Therefore a later
experimental adapter must audit every downstream acoustic-dependent path,
including narrative text and branch/routing descriptors. A raw donor bundle
cannot be relabeled or passed as a valid recipient production bundle. The
experimental payload is a separate adapter input, with original typed records
untouched. If clean isolation is not feasible without production changes, stop
for a new reviewed design; do not pretend this ablation already runs in Core.

Physics-removal is not a primary v0.1 ablation: removing the physics that supplies
the acoustic prerequisites changes both upstream information and representation
availability. A versus B covers an explicit physics comparison. A later physics
factorial experiment would require a separate feasible intervention specification.

## 12. Deterministic shuffle control

Prepare donor packets using only each donor's prefix at the same scheduled frame
index and source timestamp. Group by `(split, cutoff_frame, cutoff_timestamp)`.
Sort unique sequence IDs lexicographically within each group, and assign each
recipient the next donor cyclically. With at least two members there is no
self-assignment; singletons yield unavailable shuffle, never identity mapping.
Never cross development/test splits. No outcome or target label is used to group,
sort, match or choose donors. The sample-independent acquisition schedule is shared.

Shuffle only the acoustic payload. Do not shuffle visual observations, task
instructions, annotations or ordinary physical state. Keep donor identity,
original timestamps and original provenance in an audit sidecar unavailable to
the inference model; the recipient inference sees a packet labeled inferred
acoustic context with no donor labels. Empty but successfully generated packets
participate. Missing generation is not an empty packet and yields unavailable
slots. Register all slots before looking at outcomes and report missing/singleton
coverage. If eligible slots differ across conditions, report both per-condition
coverage and a paired common-eligibility analysis; do not silently discard trials.
Equal packet contents can survive rotation; report this limitation without
choosing a more favorable permutation based on results.

`shuffle_assignments` produces audit assignments only; it does not transform or
inject representation content. No such adapter or experimental execution is
added in Step 28A.

## 13. Dataset design and readiness gates

Stage 1 uses small controlled silent sequences with visible physical dynamics,
meaningful partial observability, objectively annotatable focal appearance,
positive and matched no-event trials, fixed source resolution, validated ordered
timestamps, and at least 10 frames/second (prospective acquisition requirement).
Require at least one scheduled decision with a full 2-second follow-up. Prefer
matched acquisition blocks sharing duration, frame rate, regions and nuisance
conditions; do not select positives only when acoustic generation succeeds.
Include confound controls where similar visible motion occurs without later
appearance. Match gross scene/camera differences without making labels available
to inference. Keep all variants from a source recording/acquisition group in one
split. Record all inclusion/exclusion decisions before model test outputs.

Stage 2 may use larger real-world benchmarks. Something-Something V2 is a
candidate only; no claim is made that its labels supply the required focal
appearance/onset truth. Benchmark suitability and supplemental annotation must
be checked separately. No data is downloaded, selected or generated here.

Before any final experiment, freeze a dataset/run manifest with sample count and
stopping rule, hashes, splits, acquisition specification, category/region manual,
annotation provenance and masks, frame schedule, exact models/prompts/adapters,
seeds, budgets, environment, ablation audits and analysis plan. Resolve missing
runner functionality on development-only fixtures, then freeze it before held-out
inference. No final outcome inspection to choose these settings. Any substantive
protocol change requires a new version before fresh held-out evaluation.

## 14. Separate endpoints and exclusions

| Endpoint | Numerator/value | Denominator/population |
| --- | --- | --- |
| `exp001.valid_anticipation_coverage` | Positives with at least one valid anticipation | Eligible positive trials with complete condition logs |
| `exp001.strict_target_event_precision` | Valid pre-event commitments | All resolved pre-event commitments on positives plus commitments on negatives |
| `exp001.tta_seconds` | Per-trial positive lead; distribution and paired differences | Positives with valid anticipation; always disclose success counts and paired selection |
| `exp001.negative_trial_false_alarm_fraction` | Negative trials with at least one false commitment | Eligible negative trials with complete logs |
| `exp001.false_commitment_fraction` | False pre-event/negative commitments | Same resolved commitment denominator as event precision |
| `exp001.false_commitment_count` | Raw number of false commitments | Count, not a ratio; disclose scheduled opportunities |
| `exp001.excluded_fraction` | Trials excluded from the endpoint | All supplied trials; stratify reason and condition |
| `exp001.prediction_log_coverage` | Recorded scheduled decisions, including abstentions | All scheduled condition cutoffs |

All empty-denominator ratios are unavailable, never zero. Valid anticipation
coverage zero is legitimate for eligible positives with complete abstention logs.
A system issuing no commitments has unavailable precision, not perfect precision.
Precision is explicitly a commitment metric, not ordinary trial accuracy.
Primary event labels are binary exact matches; no event-level partial credit is
invented. Weighted outcome score is only an optional separate Step 27C geometric
metric on its existing qualified inputs, retaining 1/0.5/0 and original labels.
Never map event matches into geometric `correct`/`partially_correct` labels.

Keep annotation exclusions separate from condition execution/log failures.
Use all supplied trials for accounting, and disclose eligible denominators for
each condition and paired common subsets. A missing/ambiguous timestamp or
annotation cannot become incorrect. Contradictory chronology, wrong sequence,
rewritten IDs, stale/future prefixes or conflicting decisions raise validation
errors and remain in a separate ingestion/violation audit. They are not silently
repaired, omitted or inserted into a score. Positive/negative differences in
exclusion rates must be reported.

No Brier, NLL or ECE: all remain **ineligible** without a separately reviewed
probability-forecast contract. No global composite or significance calculation.

## 15. Relationship to Step 27C

Step 27C remains frozen and continues to report generic TTA as
`protocol_required`. This separate Experiment 001 module specifies task-level
event matching and TTA for its own annotated records. It does not change
`EvaluationMetricsReport`, geometric accuracy, weighting, chronology checks or
probability eligibility. The producer's 0.25-second geometric matching grace is
not a tolerance for Experiment 001 appearance onset. There is no post-onset grace
for pre-event anticipation.

## 16. FlyBrain separate experimental branch

FLYBRAIN IMPLEMENTATION NOT PRESENT IN CORE. No architecture is invented and no
integration is performed. A later optional FlyBrain-inspired condition must use
the same admissible prefix, task, truth, cutoff, horizon, matching and endpoints,
with its own model/adapter manifest and distinguishable result namespace. It is
not a dependency of primary Experiment 001. Its improvement is not retroactive
evidence for Core, and its failure does not automatically falsify Core. Additional
training data or compute must be disclosed in that separate comparison.

## 17. Protocol validation implementation

`evaluation/experiment_001/protocol_v01.json` is the machine-readable declaration.
`ProtocolDefinition` canonicalizes it and assigns a SHA-256 content identity;
later runs must record this ID and the source commit, annotation/run-manifest
hashes. No wall-clock timestamp or result enters protocol identity.

`TrialAnnotation`, `PredictionWindow`, `AnticipationMatch`, `TTAResult` and
`AcousticSlot` are frozen records with detached immutable tuples and deterministic
serialization. `match_anticipation`, `assess_trial` and `shuffle_assignments` are
pure contract functions. `load_protocol()` explicitly reads the local manifest;
module import performs no I/O and imports no production modules, images or models.
`assess_trial` keeps individual matches and missing-cutoff accounting; it does
not aggregate experiment endpoints or run inference. Step 28A exercises it only
with synthetic fixtures. It cannot independently authenticate source evidence,
annotation masks, complete dataset provenance or future runner isolation.

Dedicated tests: `evaluation/tests/test_experiment_001_protocol.py`.
Validation results will be recorded after the ordered regression run.

## 18. Scientific status and next step

Protocol implemented; dedicated synthetic validation is separate from running
Experiment 001. No evidence of better hidden-state inference, acoustic benefit,
earlier reliable anticipation or baseline superiority is claimed.

Review and freeze this protocol before constructing or selecting the experimental
dataset. Then complete the dataset/run preregistration and audited adapters before
test inference. No commit, tag, push or experiment is performed in Step 28A.

PROTOCOL IMPLEMENTED. EXPERIMENT NOT YET RUN. HYPOTHESIS NOT YET SUPPORTED.
