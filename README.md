# rubik_training_cpp

Experimental C++/LibTorch port of the
[rubik_training](https://github.com/Jakov-Markic/rubik_training) Python
pipeline — same pose-model training workflow (cube bbox + 8 corner
keypoints), reimplemented from scratch in C++ to see how far a from-scratch
LibTorch build could get without Python/Ultralytics at runtime. See
**[README_CPP.md](README_CPP.md)** for the full status (what's verified vs.
not), architecture, and build instructions — this file is just the entry
point.

## Related repos

Split out of a single local working directory alongside two other projects:

- **[rubik_training](https://github.com/Jakov-Markic/rubik_training)** —
  the Python pipeline this ports, and a **build-time dependency of this
  repo** (see Prerequisites below).
- **[SolveMyCube](https://github.com/Jakov-Markic/SolveMyCube)** — the
  Flutter app that ultimately loads a trained model (via the bridge
  described in `DEPLOY_TO_APP.md`, not directly from here).

The docs in this repo reference sibling folders from that original layout —
`../rubik_training/`, `../shared_datasets/`, `../solve_my_cube/` — that
aren't part of this repo. Clone all three side by side, in a common parent
folder, for those paths to resolve as written.

## Prerequisites

- **Visual Studio 2022** (C++ workload, for `cl.exe`).
- **`rubik_training` cloned as a sibling directory, with its own venv set
  up** (see its README). This repo doesn't vendor its own CMake/Ninja/Python
  — `configure.bat` calls them from `../rubik_training/venv/Scripts`, and
  the LibTorch headers/libs this project builds against are sourced from
  that venv's `torch` install too (see `README_CPP.md`'s "Build" section for
  exactly how). If your clone lives at a different path than
  `.../rubik_training/venv`, edit the `VENV` path at the top of
  `configure.bat` / `configure_gpu.bat` to match.

## What's not in this repo

`.gitignore` deliberately strips:

- `build/` — CMake build output, regenerate with `configure.bat` + build.
- `third_party/` — vendored OpenCV + CPU-only LibTorch (large binaries, not
  source we own); see `README_CPP.md`'s "Build" section for the exact
  `pip install --target third_party/torch_cpu ...` command that recreates
  it.
- **All images/video** (`*.jpg`, `*.png`, `*.mp4`, ...) — training data
  includes images sourced from Kaggle that aren't ours to redistribute, and
  this also catches this project's own training-diagnostic plots
  (`val_predictions.jpg` etc.), which embed the dataset images.
- **All model weights/exports** (`*.pt`, `*.tflite`, `*.torchscript`,
  `*.ptl`, `*.onnx`).

## Build

```powershell
cd rubik_training_cpp
.\configure.bat                              # Phase-1 tools only (no LibTorch)
.\configure.bat -DRUBIK_BUILD_TORCH_TOOLS=ON # + LibTorch-dependent tools (training/export)
..\rubik_training\venv\Scripts\cmake.exe --build build
```

Full details — GPU build, the LibTorch-source workaround, known CMake/CUDA
compatibility issues on Windows — are in `README_CPP.md`'s "Build" section.

## Tools

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
| `train_cube_pose_8pt` | Train the 8-keypoint cube-pose model. |
| `export_cube_pose_8pt` | Export a trained checkpoint for deployment. |

Each mirrors its Python counterpart's CLI flags; run with `--help` for the
full flag list.

## Further docs

- **[README_CPP.md](README_CPP.md)** — verification status per phase,
  Phase 3 architecture (loss/assigner/augmentation), build instructions
  (CPU + GPU), and known deviations from the Python originals.
- **[GUIDE_LIBTORCH_TRAINING.md](GUIDE_LIBTORCH_TRAINING.md)** — a fuller
  walkthrough: installing LibTorch, preparing a dataset, training, and what
  it takes to get a model from here into the Flutter app.
- **[DEPLOY_TO_APP.md](DEPLOY_TO_APP.md)** — bridging a LibTorch-trained
  checkpoint into an Ultralytics-compatible export so it can reach the app.
