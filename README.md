# DINO Performance Evaluation

Benchmarking **DINOv2** against **DINOv3** for missing-component and defect
detection on PCB assemblies — measuring both **accuracy** and **latency**.

The concrete task: given a reference ("golden") image of a correctly assembled
board, decide for each screw position whether the screw is **present** or
**missing**, and find out which backbone does it better and how fast.

## How the detection works

No training and no labelled examples are needed at inference time. The pipeline
is one-shot template matching on frozen self-supervised features:

```
reference image ──► pick screw ROIs ──► embed each ROI  ─┐
                                                          ├─► cosine similarity ──► threshold ──► PRESENT / MISSING
live frame ──► SIFT homography registration ──► crop ROIs ─┘
```

1. Capture a reference image of a known-good board.
2. Click each screw position once to define its ROI.
3. For every live frame, register it onto the reference with SIFT + RANSAC
   homography, so the ROIs land in the right place even if the board shifts.
4. Embed each ROI crop with a DINO backbone.
5. Compare against the reference embedding by cosine similarity; below threshold
   means the component is missing.

Because the decision is a similarity score rather than a trained classifier,
**everything rests on how good the features are** — which is exactly what this
repo sets out to quantify.

## Status

**Live detection works.** You can run any of the four backbones against a USB
webcam and watch per-ROI similarity scores in real time.

**The evaluation harness does not exist yet** — labelled dataset capture,
accuracy scoring and latency benchmarking are the next phase. Until then this
repo demonstrates the pipeline but does not yet answer the question it was built
to answer. The design is recorded in
[the design spec](docs/superpowers/specs/2026-09-16-dinov2-v3-webcam-eval-design.md),
and the phase breakdown in
[the implementation plan](docs/superpowers/plans/2026-09-16-foundation-and-backends.md).

Notes:

- `product_config.json` and the reference image are generated per setup by
  `capture_reference.py` and `select_screws.py`, and are not committed — they are
  specific to one camera, rig and product.
- The prebuilt TensorRT engine from the original Jetson prototype will not load
  on other GPUs, so it is excluded from git. A TensorRT backend slots into
  `backends/` when the Orin work begins.

## Getting started

Requires Python 3.10, an NVIDIA GPU with CUDA 12.8, and a USB webcam.

The default camera is `USB_CAMERA_INDEX=2`. Check which index is which with
`v4l2-ctl --list-devices`, then confirm per node, since that command groups
several cameras under one bus path:

```bash
for n in 0 2 4; do echo -n "video$n: "; v4l2-ctl -d /dev/video$n --info | grep 'Card type'; done
```

V4L2 indices are assigned in enumeration order and can shift between boots or
USB ports, so re-check after replugging — a silent change would mean captures
came from a different camera than the reference.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Verify the install found CUDA:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Then set up a product and run detection:

```bash
python capture_reference.py     # SPACE saves a golden reference image, Q quits
python select_screws.py         # click each screw, S saves, Q cancels
python live_detection.py --model dinov2-small
```

`--model` accepts `dinov2-small`, `dinov2-base`, `dinov3-small` or
`dinov3-base`; `--pooling` accepts `cls` or `mean`. Press Q to quit. Every
setting in [`config.py`](config.py) can also be overridden by an environment
variable of the same name, so no script needs editing to change camera or
threshold:

```bash
USB_CAMERA_INDEX=0 MATCH_SCALE=0.5 python live_detection.py --model dinov3-small
```

### Tests

The suite runs without a camera or a GPU:

```bash
python -m pytest tests/ -m "not download"   # fast, no model downloads
python -m pytest tests/                      # also downloads the four backbones
```

## Models under evaluation

| Model | Source | Dim | Patch | Weights licence |
|---|---|---|---|---|
| DINOv2 ViT-S/14 | `facebook/dinov2-small` | 384 | 14 | Apache-2.0 |
| DINOv2 ViT-B/14 | `facebook/dinov2-base` | 768 | 14 | Apache-2.0 |
| DINOv3 ViT-S/16 | `timm/vit_small_patch16_dinov3.lvd1689m` | 384 | 16 | DINOv3 Licence |
| DINOv3 ViT-B/16 | `timm/vit_base_patch16_dinov3.lvd1689m` | 768 | 16 | DINOv3 Licence |

Feature dimensions match across generations (S = 384, B = 768), so the v2-vs-v3
comparison is like-for-like.

> **Note on access:** the `facebook/dinov3-*` repositories are gated and need
> manual approval from Meta. The `timm` mirrors above carry the same weights and
> are **not** gated, so no approval is required to reproduce these results.

## Hardware

- **Developed on:** x86_64, Ubuntu 22.04, RTX 4090 Laptop (16 GB), CUDA 12.8,
  with a USB webcam.
- **Originally built for:** Basler industrial camera on a Jetson Orin Nano.
- **Deployment target:** Jetson Orin Nano.

Both Basler and USB webcam capture are supported through a common interface in
`camera_source.py`. Feature extraction sits behind a pluggable backend so the
same evaluation can run under PyTorch on a workstation and TensorRT on the Orin.

Latency figures are only meaningful for the hardware they were measured on and
will always be reported with the device stated.

## Repository layout

```
requirements.txt        pinned dependencies (CUDA 12.8, Python 3.10)
config.py               single source of truth for runtime settings
camera_source.py        camera abstraction (Basler + USB webcam)
roi.py                  square ROI extraction with border padding
registration.py         SIFT/RANSAC homography registration
backends/               pluggable feature extractors (DINOv2, DINOv3)
capture_reference.py    capture the golden reference image
select_screws.py        click screw ROIs -> product_config.json
live_detection.py       live detection, selectable DINO backend
tests/                  hardware-free test suite
pytest.ini              pytest configuration
docs/superpowers/       design spec and implementation plan
```

## Licence

The code in this repository is licensed under the
**[Apache License 2.0](LICENSE)**.

Model weights are **not** redistributed here — they are downloaded from Hugging
Face at runtime and remain under their own licences:

- **DINOv2** — Apache-2.0.
- **DINOv3** — [Meta DINOv3 Licence](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m/blob/main/LICENSE.md),
  a custom licence, *not* Apache-2.0. If you use the DINOv3 weights, read it:
  it carries redistribution conditions, use restrictions, and an obligation to
  acknowledge DINO materials when publishing results derived from them.

## Acknowledgements

This work evaluates DINO models released by Meta AI.

```bibtex
@article{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timoth{\'e}e and Moutakanni, Th{\'e}o and others},
  journal={arXiv preprint arXiv:2304.07193},
  year={2023}
}

@article{simeoni2025dinov3,
  title={DINOv3},
  author={Sim{\'e}oni, Oriane and Vo, Huy V and Seitzer, Maximilian and Baldassarre, Federico and Oquab, Maxime and Jose, Cijo and Khalidov, Vasil and Szafraniec, Marc and Yi, Seungeun and Ramamonjisoa, Micha{\"e}l and others},
  journal={arXiv preprint arXiv:2508.10104},
  year={2025}
}
```

DINOv3 weights are accessed via [`timm`](https://github.com/huggingface/pytorch-image-models)
(Ross Wightman).

## Scope and caveats

This measures the accuracy of **this pipeline** — one-shot template matching on
frozen DINO features — not the accuracy of DINOv2 or DINOv3 in the abstract.
Results will not carry over to a fine-tuned classifier built on the same
backbone, and they are specific to the parts, optics and lighting used to
capture the dataset.
