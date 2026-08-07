# MosSongPlus P0 — Training/Deployment + Grad-CAM Correctness

Repository:
C:\Users\thant\Documents\clones\master

## Objective

Implement the confirmed P0 corrections from the completed audit.

This is implementation work, NOT another research/audit run.

Do not:
- submit another ChatGPT inspection
- repeat NotebookLM/arXiv research
- modify AgentHub, AgentHubDurable, Caveman, or MCP configuration
- retrain models
- run expensive hyperparameter searches
- expose credentials/authentication state

Use repository MCP/navigation tools only when needed.

## 1. Repository safety

Before edits:
- record git status
- record branch
- record HEAD
- preserve unrelated uncommitted work
- create branch `fix/p0-train-deploy-gradcam-alignment` when appropriate

## 2. Canonical preprocessing/input contract

Create one versioned source of truth for the deployment input contract.

Represent actual implemented behavior only.

Include where applicable:
- sample_rate_hz
- frame_length_samples
- duration
- channels
- tensor layout
- input dtype
- PCM/int16 scaling
- DC removal
- RMS/gating
- normalization method/parameters
- clipping behavior
- model input shape
- INT8 scale
- INT8 zero_point

Target flow:

PCM
→ canonical preprocessing
→ model tensor
→ TFLite quantization
→ firmware input

Fixed-shape TFLM deployment is allowed and preferred.

Do NOT make TFLM dynamically shaped merely because upstream training supports variable lengths.

## 3. Remove independent [1,2400,1] export magic value

Inspect:

src/wingbeat_ml/export/tflite.py

and callers.

Remove independently hard-coded deployment dimensions.

Resolve fixed deployment shape from:
- canonical preprocessing/model configuration; and/or
- actual model input signature

Validate agreement.

Configuration/model disagreement must fail loudly.

Do not silently truncate/pad to hide contract mismatch.

Record resolved shape in deployment/export metadata.

Add mismatch tests.

## 4. Python ↔ firmware preprocessing parity

Inspect at minimum:

src/wingbeat_ml/data/audio.py
src/wingbeat_ml/export/
deployment/tflitemicro/main/audio_provider.cc
deployment/tflitemicro/main/config.h

Determine exact Python and firmware operation sequences.

Remove unjustified divergence.

Add deterministic golden-vector testing:

raw PCM
├─ Python preprocessing → reference
└─ firmware-equivalent preprocessing → deployment reference

Floating stages:
- compare within documented tolerance

INT8 stages:
- quantize reference float using exact scale/zero_point
- compare against firmware-equivalent result

Test negative, zero, positive and saturation boundaries.

Do not claim physical-device parity unless actually tested on hardware.

## 5. Quantization contract

Export/validate:
- input dtype
- input shape
- input scale
- input zero_point
- output quantization parameters where applicable

Prevent mismatch between:

Python preprocessing range
↔ TFLite INT8 quantization
↔ firmware QuantizeInput logic

Do NOT implement the complete domain-shift/PTQ calibration framework in this goal.

## 6. Grad-CAM true class score

Inspect:

src/wingbeat_ml/evaluation/gradcam.py
src/wingbeat_ml/evaluation/diagnostics.py

Do NOT use:

log(softmax_probability)

as a replacement for raw logits.

For Dense + Softmax:

h
→ z = hW + b
→ softmax(z)

Grad-CAM target must be z_k.

If a supported architecture cannot provide a valid pre-softmax score, fail explicitly instead of silently substituting a mathematically different target.

Test this behavior.

## 7. Stable Grad-CAM batch semantics

Remove ambiguous unconditional `tf.squeeze`.

Define:

feature activations: [B,T',C]
raw CAM:            [B,T']
upsampled CAM:      [B,T]

Output rank must not depend on B.

Test:
- B=1
- B>1
- T'=1
- no scalar collapse

## 8. Raw vs display CAM

Preserve analytical magnitude.

Return equivalent structured information containing:

- raw_cam
- display_cam
- raw_min
- raw_max
- raw_mean
- raw_l1
- raw_l2
- degenerate_heatmap

Do not force raw analytical CAM through per-sample min-max normalization.

Display normalization:

range = max(raw_cam) - min(raw_cam)

if range <= 1e-8:
    display_cam = zeros
    degenerate_heatmap = true
else:
    display_cam = (raw_cam - min(raw_cam)) / range
    degenerate_heatmap = false

A constant positive CAM MUST NOT become an all-one heatmap.

## 9. Temporal alignment

When Conv1D temporal length T' differs from input T:

- explicitly align/upsample to T
- preserve batch dimension
- document interpolation method
- test final length == T

If exact receptive-field-center alignment is unavailable, document interpolation as an approximation.

## 10. Lightweight raw-CAM aggregation

Aggregation must operate on raw_cam, not independently normalized display_cam.

Provide lightweight support for:
- count
- mean
- variance/std
- median/quantiles when practical

Do not generate one plot per sample.

Do not implement full W&B logging in this goal.

## 11. Tests

Cover at minimum:

A. canonical contract
B. export shape derived from model/config
C. shape mismatch rejection
D. deterministic Python preprocessing
E. firmware-equivalent preprocessing parity
F. INT8 scale/zero_point
G. clipping/saturation boundaries
H. pre-softmax Grad-CAM target
I. B=1
J. B>1
K. T'=1
L. zero CAM
M. constant-positive CAM
N. raw/display separation
O. temporal alignment to T
P. aggregation preserves raw magnitude differences

Do not weaken existing tests.

## 12. Scope

Do not change trained model architecture or weights unless unavoidable.

Primary change areas:

- preprocessing/input contracts
- export validation
- quantization metadata/parity
- Grad-CAM diagnostics
- tests
- concise documentation

No retraining.

## 13. Verification

Run applicable repository verification:
- formatting
- lint/static checks
- unit tests
- integration tests
- export tests
- Grad-CAM tests
- project-wide verification where practical

Inspect final git diff.

No unrelated generated artifacts.

Do not complete while tests fail.

## 14. Final report

Return compactly:

1. Files changed
2. Canonical preprocessing contract
3. Training↔deployment mismatches fixed
4. Quantization validation
5. Grad-CAM corrections
6. Tests
7. Verification results
8. CONFIRMED BY TESTS
9. REQUIRES ESP32-S3 TARGET VALIDATION
10. Commit SHA

## Acceptance

Complete only when:

- independent export `[1,2400,1]` magic value is removed
- fixed deployment shape derives from canonical/model contract
- preprocessing contract is explicit
- deterministic parity tests pass
- INT8 scale/zero-point/clipping tests pass
- Grad-CAM targets true pre-softmax scores
- Grad-CAM maintains [B,T]
- constant CAM is flagged degenerate
- raw and display CAM are separate
- temporal alignment is explicit/tested
- existing behavior does not regress
- verification passes
- changes are committed

Commit message:

fix: align training deployment contract and gradcam semantics

Only after all acceptance criteria pass, emit:

<goal-complete/>
