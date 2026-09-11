# V-JEPA 2.1 Sandbox v0.1

Status: FROZEN SANDBOX MILESTONE

## Scope

This milestone validates the local V-JEPA 2.1 encoder and predictor execution path for future integration with the AISTHESIS hybrid physical world model.

No V-JEPA source code is vendored into AISTHESIS-Core.

No production AISTHESIS modules were modified.

The external sandbox used for these experiments was:

C:\Users\ecmtu\AISTHESIS-Sandbox\vjepa2

## Hardware

GPU:
- NVIDIA Quadro T2000
- 4 GB VRAM

Python:
- Python 3.11

PyTorch:
- torch 2.11.0+cu128
- CUDA 12.8
- CUDA available: True

## Model

Model:

vjepa2_1_vit_base_384

Checkpoint:

vjepa2_1_vitb_dist_vitG_384.pt

Student encoder:
- embedding dimension: 768
- parameters: 86,833,152

Predictor:
- internal predictor width: 384
- output teacher-space width: 1664
- return_all_tokens: True

The predictor output width is determined by:

vjepa2_1_teacher_embed_dim = 1664

## Upstream checkpoint URL issue

The inspected upstream repository currently contains a development override where VJEPA_BASE_URL points to localhost.

The official ViT-B checkpoint was therefore manually cached before loading.

No AISTHESIS production code was changed to work around this upstream issue.

## Encoder smoke tests

Pretrained V-JEPA 2.1 ViT-B inference completed successfully.

Examples:

Input:
(1, 3, 4, 224, 224)

Output:
(1, 392, 768)

GPU FP32 peak memory:
approximately 366 MB

Input:
(1, 3, 8, 224, 224)

Output:
(1, 784, 768)

GPU FP32 peak memory:
approximately 380 MB

Input:
(1, 3, 4, 384, 384)

Output:
(1, 1152, 768)

GPU FP32 peak memory:
approximately 402 MB

Native FP32 inference was used.

FP16 was not adopted because upstream V-JEPA 2.1 execution paths may encounter SDPA dtype / unfused-attention issues.

## Official preprocessing

The official V-JEPA 2.1 preprocessing pipeline was used.

For crop_size=384:

- resize short side to 438
- center crop 384 x 384
- convert clip to tensor
- ImageNet normalization

Output layout:

C x T x H x W

## Controlled videos

Case A:

case_a_ball_human.mp4.mp4

Verified:
- 240 frames
- 24 FPS
- 1280 x 720
- 10 seconds

Case B:

case_b_ball_no_human.mp4.mp4

The same controlled pair used in earlier AISTHESIS evaluation work was used.

## Full-video exploratory latent comparison

Uniform eight-frame sampling produced:

Case A latent:
(1, 2304, 768)

Case B latent:
(1, 2304, 768)

Pooled cosine:
0.993957

Pooled L2:
2.7092

Mean token cosine:
0.884779

This comparison is NOT used as evidence of anticipation because uniform sampling over videos of different durations maps the clips to different timestamps and allows future-event contamination.

## Aligned encoder baseline

Both clips were sampled at identical timestamps:

0.0
0.5
1.0
1.5
2.0
2.5
3.0
3.5 seconds

Equivalent frame indices at 24 FPS:

0
12
24
36
48
60
72
84

Results:

A latent:
(1, 2304, 768)

B latent:
(1, 2304, 768)

Pooled cosine:
0.996929

Pooled L2:
1.946222

Mean token cosine:
0.928745

Temporal cosine similarity:

0.0-0.5 s:
0.993758

1.0-1.5 s:
0.993847

2.0-2.5 s:
0.996386

3.0-3.5 s:
0.995771

## Interpretation

No monotonic pre-event latent divergence was observed before the first visible human event.

Encoder similarity alone therefore provides no evidence of hidden-human anticipation in this controlled pair.

Latent divergence must not be interpreted as actor detection or physical fact.

## Causal predictor mask validation

For an eight-frame 384 x 384 clip:

Spatial tokens per temporal step:

24 x 24 = 576

Tubelet size:

2

Temporal token steps:

4

Total tokens:

2304

Context mask:

0..1727

Shape:

(1, 1728)

Target mask:

1728..2303

Shape:

(1, 576)

This corresponds to:

temporal step 0 -> tokens 0..575
temporal step 1 -> tokens 576..1151
temporal step 2 -> tokens 1152..1727
temporal step 3 -> tokens 1728..2303

Masked encoder execution succeeded.

Context latent:

(1, 1728, 768)

Predictor execution succeeded.

Predicted target:

(1, 576, 1664)

## Predictor target-space discovery

The student encoder produces 768-dimensional representations.

The released distilled predictor produces 1664-dimensional target representations.

Therefore student full-clip output must NOT be used directly as the predictor ground truth.

No truncation, zero-padding, random projection, or cross-space comparison was used.

## Distillation target

Repository and checkpoint inspection established that the distilled ViT-B predictor targets the 1664-dimensional representation space of a V-JEPA 2.1 ViT-G teacher.

Teacher architecture:

vit_gigantic_xformers

Properties:

- embedding dimension: 1664
- transformer blocks: 48
- attention heads: 26
- patch size: 16
- tubelet size: 2
- final block index: 47

The ViT-B checkpoint contains:

- encoder
- ema_encoder
- predictor
- optimizer/scaler/training metadata

It does NOT contain:

- target_encoder
- teacher weights

## Required teacher checkpoint

Official teacher checkpoint:

vjepa2_1_vitG_384.pt

Official public checkpoint URL:

https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt

A HEAD-only HTTP request reported:

30,238,058,912 bytes

approximately:

30.24 GB
28.16 GiB

The checkpoint body was NOT downloaded.

## Teacher hardware assessment

Teacher encoder parameters:

1,845,216,768

Approximate parameter storage:

FP32:
6.87 GiB

FP16 / BF16:
3.44 GiB

The NVIDIA Quadro T2000 4 GB is not considered a reliable target for full ViT-G teacher inference because runtime memory is required in addition to model weights.

CPU-only encoder inference is the preferred future evaluation route for this hardware.

The machine has approximately 32 GiB system RAM.

## Correct future predictor evaluation

A faithful future predictor evaluation requires:

Context clip
    ->
ViT-B student encoder
    ->
768-dimensional context representation
    ->
V-JEPA predictor
    ->
1664-dimensional predicted teacher-space target

and independently:

Complete clip
    ->
frozen ViT-G teacher
    ->
1664-dimensional final-layer representation
    ->
training-equivalent normalization
    ->
apply target mask
    ->
1664-dimensional actual teacher target

Only then may prediction error be computed.

The teacher full-clip branch is retrospective reference information only.

Teacher features must never be fed into the causal context branch.

## Deferred work

Exact teacher-target predictor evaluation is intentionally deferred.

Reason:

The required V-JEPA 2.1 ViT-G checkpoint is approximately 30.24 GB and was not downloaded during this milestone.

This does not block using the validated ViT-B encoder as a learned latent representation source for the next AISTHESIS hybrid world-model stage.

## Scientific constraints

The following rules remain mandatory:

- prediction != observation
- latent representation != physical fact
- latent divergence != hidden actor detection
- possibility != fact
- expected consequence != observed consequence
- missing evidence != contradiction
- no future leakage
- no fabricated hidden actors
- no cross-space latent comparison

## Milestone conclusion

V-JEPA 2.1 ViT-B encoder and predictor execution paths were successfully validated locally.

The following are verified:

- pretrained encoder loading
- official preprocessing
- GPU inference
- controlled Case A / Case B encoding
- aligned latent comparison
- causal context masking
- causal target masking
- masked student encoding
- predictor execution
- 1664-dimensional teacher-space prediction output

Exact teacher-target prediction accuracy remains deferred until the external V-JEPA 2.1 ViT-G teacher checkpoint is deliberately acquired.

Milestone:

V-JEPA 2.1 Sandbox v0.1 — Encoder & Predictor Path Validated
