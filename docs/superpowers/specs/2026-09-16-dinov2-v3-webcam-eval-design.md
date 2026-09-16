# DINOv2 vs DINOv3 Defect-Detection Evaluation — Design

**Date:** 2026-09-16
**Status:** Approved, pending implementation plan

## Goal

Measure the **accuracy** and **latency** of DINOv2 and DINOv3 backbones on
screw-presence / missing-component detection, using a USB webcam in place of the
Basler camera the repo was originally built around.

## Scope Decisions

| Decision | Choice |
|---|---|
| Accuracy method | Labeled dataset + offline scoring harness |
| Latency target | This laptop (RTX 4090) now; Orin Nano later via pluggable backend |
| Models | dinov2-small, dinov2-base, dinov3-small, dinov3-base |
| Test object | PCB with 4-5 physically removable screws |
| Capture protocol | Pose + lighting variation |
| Code structure | Shared modules + thin scripts |

## Environment (verified 2026-09-16)

- x86_64, Ubuntu 22.04, kernel 6.8, 28-core i7-13850HX, 31 GB RAM
- NVIDIA RTX 4090 Laptop, 16 GB VRAM, driver 570.211.01, CUDA 12.8
- Webcam: `Integrated_Webcam_FHD` at `/dev/video0`
- Python 3.10.12; **no** torch / transformers / opencv installed anywhere
- HuggingFace reachable; token present for user `darylfoo`

## Starting-State Problems

These are pre-existing and must be resolved as part of the work:

1. **`trt_feature_extractor.py` is missing.** `live_detection_trt.py:9` imports it;
   the file does not exist. The main script cannot run at all.
2. **`dinov2_small_features_fp16.engine` is unusable here.** Built for Jetson Orin
   Nano (TensorRT 10.3, SM 8.7). TensorRT engines are not portable across GPU
   architecture or TRT version. This machine is SM 8.9.
3. **`product_config.json` and `my_photo-1.jpg` are absent.** Both are required
   inputs to `live_detection_trt.py`.
4. **No accuracy measurement exists.** The repo has no ground truth, no labels,
   and no metrics — only a live visual overlay.
5. **Constants are duplicated across three scripts.** `CAMERA_BACKEND`,
   `ROTATE_180`, `USB_CAMERA_INDEX`, frame size and Basler settings are
   copy-pasted into `capture_reference.py`, `select_screws.py` and
   `live_detection_trt.py`. All three hardcode `CAMERA_BACKEND = "basler"`.
6. **Latency numbers are not trustworthy.** The DINO timer at
   `live_detection_trt.py:401-408` has no `torch.cuda.synchronize()`, so it times
   kernel *launch* rather than execution.

Note that **USB webcam capture already works** — `UsbCamera` in
`camera_source.py:4-21` is wired through `create_camera`. The webcam work is
configuration plumbing plus robustness fixes, not new capture code.

## Model Access

DINOv3 under `facebook/*` is `gated: manual` and the available token returns
**HTTP 403**. The **timm mirrors are ungated and confirmed downloadable**:

| Model | Source | Dim | Patch | Input |
|---|---|---|---|---|
| dinov2-small | `facebook/dinov2-small` | 384 | 14 | processor-governed (224) |
| dinov2-base | `facebook/dinov2-base` | 768 | 14 | processor-governed (224) |
| dinov3-small | `timm/vit_small_patch16_dinov3.lvd1689m` | 384 | 16 | 256 |
| dinov3-base | `timm/vit_base_patch16_dinov3.lvd1689m` | 768 | 16 | 256 |

Feature dimensions match across generations (S=384, B=768), so v2-vs-v3 is a true
like-for-like comparison. Requesting Meta's approval is optional, not blocking.

## Architecture

```
config.py              single source of truth for all runtime constants
camera_source.py       [modified] webcam robustness + device discovery
roi.py                 [extracted] crop_square_with_padding, shared
registration.py        [extracted] SIFT / FLANN / homography, shared
backends/
  base.py              FeatureExtractor protocol
  torch_backend.py     dinov2 via transformers, dinov3 via timm
  __init__.py          get_backend(name)
capture_dataset.py     [new] guided labeled capture
preprocess.py          [new] register + crop once -> shared crop bank
evaluate.py            [new] accuracy metrics over identical crops
benchmark.py           [new] latency metrics
report.py              [new] comparison tables + plots
live_detection.py      [rewrite] --model flag, webcam default
requirements.txt       [new]
```

The `backends/` seam is what enables the later Orin Nano port: a
`tensorrt_backend.py` drops in beside `torch_backend.py` without touching any
eval code. It is also the correct home for the missing `trt_feature_extractor.py`,
turning that gap into a deliberate empty socket rather than a broken import.

### Backend interface

```python
class FeatureExtractor(Protocol):
    name: str
    dim: int
    def embed(self, crops_bgr: list[np.ndarray], pooling: str) -> np.ndarray:
        """Returns [N, D], L2-normalized so cosine similarity == dot product."""
    def warmup(self, batch_size: int) -> None: ...
```

`pooling` is `"cls"` or `"mean"`. `"mean"` averages **patch tokens only**:
for DINOv2 this excludes the CLS token; for DINOv3 it excludes CLS *and* the
register tokens. Neither generation may average a prefix token into the result.

## Data Flow

1. **Capture** — `capture_dataset.py` reads the screw list, then prompts through
   each state: `"Set board to: ALL PRESENT — lighting A"`. Operator presses a key;
   the tool records `FRAMES_PER_BURST` frames (default 40) while the board is
   gently tilted and shifted. Then
   `"Remove screw 1"`, and so on. Writes `frames/*.jpg` plus `labels.json`
   containing per-frame, per-ROI ground truth and a `burst_id`.
2. **Preprocess** — `preprocess.py` runs SIFT registration over every frame
   **once**, warps to the reference, crops all ROIs, caches to `crops/`, and
   records per-frame registration success.
3. **Evaluate** — every backend embeds the **same cached crops**. Cosine
   similarity against the reference-crop embedding yields a score per ROI, swept
   across thresholds.
4. **Benchmark** — a separate run with no disk I/O inside the timed region.

Registering once rather than inside each model's loop makes the comparison
byte-identical across models, makes evaluation ~4x faster, and lets registration
success be reported as its own pipeline metric instead of silently contaminating
model accuracy.

### Dataset layout

```
datasets/<name>/
  frames/*.jpg
  labels.json        per-frame, per-ROI ground truth + burst_id
  crops/             registered ROI crops produced by preprocess.py
  meta.json          camera, resolution, capture settings, git hash
results/<name>/
  metrics.json  latency.json  report.md  plots/
```

### Expected dataset size

The board has **4-5 screws** (`N`). Nothing hardcodes the count: the capture
tool reads `N` from `product_config.json` and generates `N + 1` board configs
(all-present, plus each screw removed in turn).

Frames = `(N + 1) x LIGHTING_CONDITIONS x FRAMES_PER_BURST`, and every frame
labels all `N` ROIs at once, so ROI samples = `N x frames`.

| N screws | Configs | Frames | ROI samples | Missing | Present |
|---|---|---|---|---|---|
| 4 | 5 | 600 | 2,400 | 480 | 1,920 |
| 5 | 6 | 720 | 3,600 | 600 | 3,000 |

(at `LIGHTING_CONDITIONS = 3`, `FRAMES_PER_BURST = 40`)

Either size gives enough missing-class samples for tight AUROC confidence
intervals. Roughly 20-35 minutes of hands-on capture.

## Metrics

### Accuracy (`evaluate.py`)

Headline metric is **AUROC** — threshold-free, so it measures whether the
features separate present from missing without conflating that with a poorly
chosen threshold. Reported per model x pooling:

- AUROC and average precision
- Best-F1 threshold, plus accuracy / precision / recall / F1 at it
- Accuracy at the repo's existing fixed `0.75` threshold, for continuity
- **d-prime separation margin** — headroom before threshold drift causes failures
- Per-ROI breakdown (some screw positions will be materially harder)
- Bootstrap 95% CI on AUROC, **resampled by burst, not by frame**

The burst-level resampling matters: frames within a single burst are highly
correlated, and resampling individual frames would report falsely tight
confidence intervals.

Registration success rate is reported **once**, model-independent.

### Latency (`benchmark.py`)

- 20 warmup iterations, then 200 measured
- `torch.cuda.synchronize()` around every timed region
- p50 / p95 / p99, not mean alone
- Batch size = number of screws (realistic) and batch size = 1
- Preprocessing (resize/normalize) timed separately from forward pass
- Peak VRAM via `torch.cuda.max_memory_allocated`
- SIFT registration timed separately — at `MATCH_SCALE=0.25` on 1080p it may
  well dominate DINO inference, which would change the optimization target

### Report (`report.py`)

Markdown table of model x pooling -> AUROC, F1, best threshold, p50 latency,
VRAM. Plots: present-vs-missing score distributions per model, ROC curves, and an
accuracy-vs-latency scatter showing the Pareto frontier — the actual decision chart.

## Correctness Details

1. **Each model uses its own preprocessing.** DINOv2 is patch14 with
   processor-governed 224px input; DINOv3 is patch16 at 256px. The current code
   hardcodes `AutoImageProcessor.from_pretrained("facebook/dinov2-small")`;
   reusing that for v3 would handicap v3 and invalidate the comparison.
2. **DINOv3 has register tokens.** Mean-pooling must skip them via timm's
   `num_prefix_tokens`, or garbage is averaged into every embedding.
3. **All timed regions synchronize CUDA.** Without it, measurements reflect
   kernel launch rather than execution.
4. **Embeddings are L2-normalized** at the backend boundary, so cosine similarity
   reduces to a dot product and no caller can forget to normalize.

## Webcam Specifics

- Set **MJPG fourcc** — UVC webcams commonly fall back to ~5 fps at 1080p on raw
  YUYV, which would distort every latency measurement
- Verify the resolution actually granted, not the one requested (webcams fall
  back silently)
- Flush the first few frames while auto-exposure settles
- Fix the `isOpened()` check, which currently runs *after* the property sets
  (`camera_source.py:7-13`)
- Default `ROTATE_180 = False` — the 180-degree rotation was a Basler mounting
  artifact, not a property of the scene

## Error Handling

- Camera open failure lists available devices and their real supported modes
- Missing `product_config.json` directs the user to run `select_screws.py`
- Missing reference image directs the user to run `capture_reference.py`
- DINOv3 download failure explains the gate and points at the timm mirror
- Label/frame mismatch fails loudly at preprocess, never silently
- Registration failure retains the existing stale/debounce behavior in the live path

## Testing

The harness must be testable **without hardware**:

- Unit: `crop_square_with_padding` edge cases — ROI at the border, partially
  outside, fully outside
- Unit: backend output shape and L2 norm == 1 for every backend
- Unit: metric functions checked against scikit-learn on synthetic scores
- Unit: burst-level bootstrap produces wider CIs than frame-level on correlated
  synthetic input (guards the statistical subtlety above)
- Golden: registration recovers a **known** homography applied to the reference,
  within tolerance
- Smoke: a synthetic dataset generator drives capture -> preprocess -> evaluate ->
  benchmark -> report end to end with no camera attached

## Implementation Phases

Each phase is independently verifiable, so the work can stop at any boundary and
still leave the repo in a working state.

1. **Foundation** — `requirements.txt`, environment install, `config.py`,
   `roi.py`, `registration.py`, webcam fixes in `camera_source.py`. Verified by
   the unit tests for cropping and the known-homography registration test.
2. **Restore the existing workflow on webcam** — rewire `capture_reference.py`
   and `select_screws.py` to `config.py`, capture a real reference of the PCB,
   and produce `product_config.json`. Verified by a reference image and config
   existing on disk.
3. **Backends** — `backends/base.py`, `backends/torch_backend.py`, all four
   models loading. Verified by shape and L2-norm tests for each backend.
4. **Live detection on webcam** — `live_detection.py` replacing
   `live_detection_trt.py`, with `--model`. This is the first point where the
   original demo runs again on this machine.
5. **Dataset capture** — `capture_dataset.py`, then an actual capture session
   with the PCB.
6. **Eval and benchmark** — `preprocess.py`, `evaluate.py`, `benchmark.py`,
   `report.py`, plus the synthetic end-to-end smoke test.

Phases 1-4 depend on no physical capture session and can be completed and
verified before the PCB is in hand; phase 5 is the only one gated on hardware
availability.

## Out of Scope

- TensorRT engine rebuilds (deferred to the Orin Nano phase; the backend seam is
  the extension point)
- Fine-tuning any backbone
- Multi-frame reference prototypes (single golden reference matches current
  deployment; revisit if separation margins prove marginal)
- Basler hardware support is retained, not extended

## Caveat to State in the Report

This measures the accuracy of **this pipeline** — one-shot template matching on
frozen DINO features — not "DINOv2's accuracy" in the abstract. It is the right
target for this use case, but the result will not generalize to a fine-tuned
classifier on the same backbone. The report must say so plainly so the numbers
are not over-read.
