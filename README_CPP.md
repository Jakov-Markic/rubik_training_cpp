# rubik_training_cpp

C++/LibTorch port of the `rubik_training/` Python pipeline. See
`../rubik_training/README_TRAINING.md` for the pipeline itself (what each
stage does and why) — this file only covers building and running the C++
tools, plus the handful of places where the port deliberately differs from
the Python originals.

Dataset folders (raw CVAT sources + merged/YOLO outputs) live in
`../shared_datasets/`, a sibling of `rubik_training/` and this directory —
both pipelines read/write the same data there. See `GUIDE_LIBTORCH_TRAINING.md`
for the full walkthrough (LibTorch setup, dataset prep, training, and why a
LibTorch-trained model can't yet be deployed to the Flutter app).

## Status

- **Phase 1 (done)**: 7 non-ML tools — `extract_video_frames`,
  `dedupe_similar_frames`, `yolo_pose_to_coco_keypoints`,
  `coco_keypoints_to_yolo_pose`, `merge_coco_keypoints_datasets`,
  `check_keypoint_label_consistency`, `visualize_keypoint_labels`. Verified
  against the Python originals on real repo data — byte-identical or
  numerically-identical output (see "Known deviations" for the one
  documented exception).
- **Phase 2 (done)**: `pseudo_label_real_cube_images` — LibTorch inference
  against a TorchScript export (`train_cube_pose_8pt.py --export-only`'s
  output). Verified two ways: (1) loading the exact same `.torchscript` file
  directly in Python with matching preprocessing produces bit-identical raw
  output to the C++ tool, confirming the letterbox/tensor-packing/decode
  pipeline is correct; (2) end-to-end predictions are in the same ballpark
  as `pseudo_label_real_cube_images.py`'s output (not identical — the
  Python script runs the *raw* checkpoint through ultralytics' own
  pipeline at a flexible `--imgsz`, e.g. 640, with proper NMS, while this
  tool runs the *wrapped, 320-fixed* export — different model
  configuration, not a bug; see the plan doc).
- **Phase 3 (done)**: `train_cube_pose_8pt` / `export_cube_pose_8pt` — a
  from-scratch LibTorch YOLOv8n-pose-shaped training pipeline, plus
  automatic post-training evaluation graphs and GPU support. Verified
  end-to-end on the real dataset (clean rebuild, both CPU and GPU): trains
  without crashing, loss decreases over epochs, checkpoints save/reload,
  all 7 graph files generate correctly. See "Phase 3 architecture" below
  and the plan doc for the full design and its explicit scope caveats
  (training-quality and mobile-deployment parity are not promised — see
  "Deployment bridge").

## Phase 3 architecture

`model/include/rubik/`:

- `modules.hpp` / `modules.cpp` — `Conv` (conv+BN+SiLU), `Bottleneck`,
  `C2f`, `SPPF`, `DFL`: the same building blocks yolov8n-pose.yaml uses.
- `yolo_pose_8pt.hpp` / `.cpp` — the assembled backbone/neck/head at the
  n-scale channel widths (16/32/64/128/256), plus `make_anchors`/
  `dist2bbox`/`flatten_levels` helpers shared with the loss.
- `augment.hpp` / `.cpp` — letterbox, random affine (rotate/translate/
  scale/shear), HSV jitter, random erasing, mosaic (see below).
- `dataset.hpp` / `.cpp` — `PoseDataset` loads a YOLO-pose dataset
  directory (as built by `coco_keypoints_to_yolo_pose`) and produces
  augmented `{images, targets}` batches directly (no `torch::data`
  `DataLoader` — mosaic needs cross-sample access that doesn't fit its
  `Dataset`/`Stack` collation model cleanly).
- `plotting.hpp` / `.cpp` — small OpenCV-drawn line/histogram/bar-chart
  utilities (no matplotlib in C++).
- `evaluate.hpp` / `.cpp` — runs the trained model over a dataset with no
  augmentation/gradient and decodes each image's single highest-confidence
  anchor (the same convention `pseudo_label_real_cube_images` and the
  Flutter app's decode both use), producing per-sample confidence/IoU/
  keypoint-error data the training tool turns into graphs after training
  finishes (`results.png`, `confidence_hist.png`, `iou_hist.png`,
  `pr_f1_curve.png`, `keypoint_error.png`, `val_predictions.jpg` — written to
  `runs_cpp/pose/<name>/`, alongside `weights/`; see `graphs_readme.txt`
  written into the same folder, or `GUIDE_LIBTORCH_TRAINING.md`, for what
  each one shows).
- `loss.hpp` / `.cpp` — CIoU box + DFL + BCE cls + OKS-style keypoint loss,
  with a **single-GT-per-image simplified assigner**: every image here has
  exactly one cube, so there's no cross-GT competition to resolve the way
  ultralytics' full task-aligned assigner does. Candidate anchors = grid
  points inside the GT box; top-k by `cls_score^0.5 * iou^6` become
  positives. cls uses hard binary targets (positive=1/negative=0) rather
  than ultralytics' soft alignment-metric-scaled targets — simpler, lower
  bug-risk, a deliberate trade against exact parity.

Augmentation scope, matching the CLI knobs `train_cube_pose_8pt.py`
exposes: rotate/translate/scale/shear affine, HSV jitter, random erasing,
and mosaic are implemented. `--perspective` is accepted but currently a
no-op (its current default, `0.0005`, is already near-zero). `--mixup` is
accepted but unimplemented (current training config never sets it nonzero,
so this is a documented no-op, not a gap that matters today). Mosaic keeps
exactly one object per composited sample (picks one of the 4 sub-images'
object as the target) rather than a real multi-instance composite, to stay
consistent with the loss's single-GT assumption.

**No pretrained-weight loading**: unlike `train_cube_pose_8pt.py` (which
fine-tunes from COCO-pretrained `yolov8n-pose.pt`), this trains from random
initialization — ultralytics' `.pt` checkpoints are pickled Python objects
that don't map onto this independently-built module graph. Expect slower/
weaker convergence than the Python pipeline on the same small dataset;
that gap is inherent to training from scratch, not a bug to fix.

### Deployment bridge

`export_cube_pose_8pt --output` writes a **LibTorch-native archive**
(`torch::save`), loadable only via `torch::load()` back into a
`rubik::YoloPose8pt` instance in this codebase — not a TorchScript file, so
it can't be consumed by Python/`ultralytics`/`colab_export_tflite.py` on
its own.

**A real bridge now exists**: `export_cube_pose_8pt --pickle-out` writes the
full state (parameters + BatchNorm running buffers) as a Python-loadable
pickle (`torch::pickle_save`, own module names), and
`tools/bridge_to_ultralytics.py` copies every tensor into a real
`ultralytics.YOLO` instance using a verified module-name correspondence
table — see `DEPLOY_TO_APP.md` for the full walkthrough and exact commands.

That correspondence table was derived by actually diffing this
architecture against a real trained ultralytics checkpoint layer-by-layer
(not from memory/assumption): the backbone+neck's 135 parameters matched
**positionally with zero shape mismatches**; the head (box/cls/kpt
branches, 3 FPN levels each) matched by explicit group once the right
`kpt_shape=[8,3]` reference checkpoint was used (an earlier attempt against
the *default* `yolov8n-pose.yaml`, which defaults to COCO's 17-keypoint
person config, produced a red herring 51-vs-24-channel "mismatch" that
wasn't real — see the git history / conversation for that dead end if it
recurs).

**Verification status — read this before trusting it blindly**: the
`load_state_dict(strict=True)` succeeds (every one of 397 tensors mapped,
zero shape errors) and predictions from a bridged checkpoint are
*correlated* with this codebase's own decode on the same images (same
rough confidence range, same rough box region) — not the random/unrelated
output you'd get from a genuinely wrong mapping. But they are **not
bit-identical**, tested on a barely-trained 3-epoch smoke checkpoint. Most
likely cause: LibTorch (C++) and PyTorch (Python) use different CPU
floating-point kernels, and small per-layer differences compound across
~40 conv layers; on a barely-converged model where many anchors have
similar low confidence, that noise can flip which anchor wins the
single-best-detection argmax, making that one prediction look very
different even though the underlying weights are identical. This is
**strong structural evidence, not a bit-for-bit mathematical proof** — the
real test is re-running `--verify-images` after an actual full training
run, where a converged model's correct anchor should dominate clearly
enough that this kind of noise stops mattering, and where you can just
look at whether the box lands on the cube.

## Build

Requires: Visual Studio 2022 (C++ workload, for `cl.exe`) and CMake+Ninja
(sourced from `../rubik_training/venv`, no separate install). C++ standard
is 20 (LibTorch 2.13's headers require it).

```powershell
cd rubik_training_cpp
.\configure.bat                              # Phase-1 tools only
.\configure.bat -DRUBIK_BUILD_TORCH_TOOLS=ON  # + LibTorch-dependent tools (Phase 2/3)
```

**LibTorch source**: `../rubik_training/venv` has `torch==2.5.1+cu121` (used
by the Python scripts), but its bundled `Caffe2Config.cmake` unconditionally
requires a full CUDA Toolkit to be findable by CMake, even for a CPU-only
C++ build — and no CUDA Toolkit is installed here. Rather than pull in a
multi-GB toolkit, `RUBIK_BUILD_TORCH_TOOLS=ON` vendors a separate CPU-only
torch wheel into `third_party/torch_cpu/` purely for its LibTorch
headers/libs/cmake package:

```powershell
..\rubik_training\venv\Scripts\python.exe -m pip install --no-deps `
  --target third_party\torch_cpu --index-url https://download.pytorch.org/whl/cpu torch
```

This never touches the venv's GPU torch install — TorchScript files
exported by the Python (cu121) side load fine here regardless, since
TorchScript is forward/backward compatible across torch versions.

### GPU build

With the CUDA 12.1 Toolkit installed (nvcc + libraries, **not** the bundled
display driver — see `GUIDE_LIBTORCH_TRAINING.md`), `configure_gpu.bat`
builds against the venv's actual `cu121` torch instead of the CPU-only
vendored copy, so `--device 0` uses the GPU. Getting there needs three
workarounds baked into that script (each is a genuine compatibility gap on
this machine, not a hack of convenience):

1. **8.3 short paths** (`C:/Progra~1/NVIDIA~2/CUDA/v12.1`) for the CUDA
   Toolkit directory — a CMake `FindCUDA.cmake` bug (policy `CMP0219`)
   mis-escapes backslashes in paths containing spaces (`Program Files`),
   breaking `find_program` calls inside it.
2. **`-allow-unsupported-compiler`** — CUDA 12.1's `nvcc` (Feb 2023) hard-
   rejects MSVC versions newer than what it shipped validated against, and
   this machine's VS2022 (17.14) postdates it by ~2 years. Safe here because
   the project has zero `.cu` files of its own — LibTorch's CUDA kernels are
   already precompiled, so this flag only unblocks CMake's *compiler
   detection* step, never actual nvcc codegen.
3. **`-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH`** — Microsoft's own
   official escape hatch for the matching STL-side static_assert (MSVC's
   `<yvals_core.h>` independently requires CUDA ≥ 12.4 for this MSVC
   version).

There's also a `CUDA::nvToolsExt` (NVTX profiling markers) target that
newer CMake's `FindCUDAToolkit` doesn't create for this CUDA/CMake
combination; `CMakeLists.txt` stands in a no-op `INTERFACE IMPORTED` target
for it — we don't use NVTX profiling markers, so this is inert, not a
missing-feature workaround.

**Performance note**: two optimizations landed after the GPU build first
came up:

1. `loss.cpp`'s per-image assignment loop used to call `.item<double>()`
   ~28 times per image (4 for the GT box, 3×8 for the GT keypoints) to pull
   values off the GPU one scalar at a time — each call forces a blocking
   GPU→CPU sync, and at batch=8 that's ~224 blocking round-trips per
   training step. Fixed by copying `targets` to CPU once per batch and
   reading every scalar from that copy instead. The one sync that's
   inherent to the algorithm (`torch::nonzero()`, since CUDA has to know
   the match count to size its output) is unavoidable without a
   fundamentally different padded/masked assignment scheme, and was left
   as-is — one sync per image rather than ~28.
2. `make_anchors()` was rebuilding the same anchor-point/stride tensors
   from scratch on every single forward pass, even though they're constant
   for a fixed input resolution. Now cached by (grid shapes, device,
   offset).

Measured impact (imgsz=320, batch=8, 15 epochs, averaging out fixed
per-run startup cost): GPU went from ~18–19.6s/epoch to **16.7s/epoch**,
CPU from ~26.3s/epoch to **23.0s/epoch** — both got ~10–15% faster, but the
GPU/CPU *ratio* barely moved (~1.4x throughout). That's the tell that the
sync fix, while real, was never the dominant cost: something that scales
the same regardless of `--device` still dominates. That's almost certainly
**data loading** — `dataset.cpp`'s image decode + letterbox + affine warp +
HSV jitter all run synchronously on CPU/OpenCV with no overlap, so the GPU
sits idle while each batch is prepared. For a model this small at 320px,
CPU prep time is comparable to GPU compute time. At the real training
resolution (640px = 4x the pixels), GPU compute per batch grows much
faster than data-loading cost, so the achievable speedup should be
meaningfully better there than this 320px benchmark suggests — a
production-settings timing check (imgsz=640, batch=8) measured **~19s/epoch
on GPU**, consistent with that expectation. Overlapping data loading with
GPU compute (a prefetch thread) would close most of the remaining gap but
hasn't been done — a real next step if training throughput matters more
than what's here now. cuDNN is also not installed (separate NVIDIA
download), so convolutions run without its optimized kernels; a smaller
effect than the above given how small this model is.

Then build. Plain `cmake` isn't on `PATH` on this machine (no system
install, only the venv's vendored copy), so use its full path:

```powershell
..\rubik_training\venv\Scripts\cmake.exe --build build
```

Executables and their runtime DLLs (OpenCV, and Torch when enabled) land in
`build/`.

## Tools

Each tool mirrors its Python counterpart's CLI flags (see
`README_TRAINING.md` for the pipeline walkthrough); run with `--help` for
the full flag list. One-line purpose per tool:

| Tool | Purpose |
|---|---|
| `extract_video_frames` | Sample sharp, non-duplicate frames from a video clip. |
| `dedupe_similar_frames` | Thin near-duplicate frames from an extracted set. |
| `pseudo_label_real_cube_images` | Pre-label frames with a trained model, for faster CVAT review. |
| `yolo_pose_to_coco_keypoints` | Convert YOLO-pose pseudo-labels to a CVAT-importable COCO JSON. |
| `merge_coco_keypoints_datasets` | Merge multiple CVAT COCO-keypoints exports into one dataset. |
| `coco_keypoints_to_yolo_pose` | Convert a merged COCO-keypoints dataset to YOLO-pose train/val. |
| `check_keypoint_label_consistency` | Flag self-intersecting (mislabeled) corner quads. |
| `visualize_keypoint_labels` | Render a contact-sheet of labeled corners for visual QA. |
| `train_cube_pose_8pt` | Train the 8-keypoint cube-pose model (LibTorch, Phase 3). |
| `export_cube_pose_8pt` | Export a trained checkpoint for deployment (LibTorch, Phase 3). |

## Known deviations from the Python scripts

- **`coco_keypoints_to_yolo_pose`'s train/val shuffle** uses `std::mt19937`
  seeded with `--seed`, not Python's Mersenne-Twister-based
  `random.shuffle`. Same split *size* and split *determinism* (same seed →
  same split every run), but not the same split *membership* as the Python
  tool given identical input and seed. Verified on the real dataset (183
  images): both languages agree on the 164/19 train/val split *counts*,
  and per-sample numeric values (box/keypoint coordinates) match exactly
  — only which samples land in train vs. val differs.
- All other Phase-1 tools produced byte-identical (image outputs) or
  structurally-identical (JSON outputs, compared as parsed Python objects)
  results against the originals when run on the same real repo data during
  development.
- **`pseudo_label_real_cube_images`** consumes a TorchScript export (fixed
  input size, single-best-detection already picked) rather than the raw
  ultralytics checkpoint the Python script uses (flexible `--imgsz`,
  ultralytics' own NMS). `--imgsz` here must match whatever `--mobile-imgsz`
  the export was traced at (320 by default) — passing a mismatched value
  will silently produce garbage coordinates, not an error.
