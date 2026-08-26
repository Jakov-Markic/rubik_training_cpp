# Guide: LibTorch setup, dataset prep, training, and getting a model into the app

A step-by-step walkthrough for someone starting from scratch: installing
LibTorch, building `rubik_training_cpp`, preparing a labeled dataset, training
the C++ pose model, and — this is the part to read carefully — what actually
happens (and doesn't yet happen) when you try to get a trained model into the
Flutter app.

If you just want to train/deploy a model that works in the app **today**,
skip to [Section 6](#6-getting-a-model-into-the-flutter-app-read-this-first).
The short version: use the existing Python pipeline in `rubik_training/` for
that; the LibTorch pipeline here doesn't have a deployment path yet.

## 1. What you're setting up

- **LibTorch**: the C++ distribution of PyTorch — gives you `torch::Tensor`,
  `torch::nn`, autograd, optimizers, etc. from C++, no Python involved.
- **This project's build**: CMake + Ninja + MSVC compiling
  `rubik_training_cpp/` against LibTorch and OpenCV.
- **A dataset**: images of the cube with labeled 8 corner keypoints, in the
  YOLO-pose format the training tool reads.

## 2. Installing LibTorch

You have two options. Pick one.

### Option A — Official prebuilt LibTorch (the standard way)

1. Go to <https://pytorch.org/get-started/locally/>.
2. Select: PyTorch build = Stable, OS = Windows, Package = **LibTorch**,
   Language = C++/Java, Compute Platform = CPU or a CUDA version matching a
   CUDA Toolkit you have installed.
3. Download the zip (there's a Release and a Debug variant — get Release
   unless you specifically need debug symbols) and unzip it somewhere, e.g.
   `C:\libtorch`.
4. Point CMake at it:
   ```powershell
   cmake -B build -DCMAKE_PREFIX_PATH=C:\libtorch
   ```
   (or set `Torch_DIR=C:\libtorch\share\cmake\Torch` directly).

**Windows ABI note**: the prebuilt LibTorch must match your compiler's
runtime (MSVC 2022 here). Mixing a Debug-CRT LibTorch with a Release build of
your own code (or vice versa) causes hard-to-diagnose linker/runtime errors —
match Release-to-Release.

If you pick the CUDA build, you need a matching **CUDA Toolkit** installed
separately (not just a GPU driver) for CMake's `find_package(CUDA)` to
succeed — this is a multi-GB download from NVIDIA. If you don't already have
one, use the CPU build instead; it's enough for inference and for training
this small a dataset (slower per-epoch than GPU, but the dataset here is only
~100 images, so CPU training is workable, just not fast).

### Option B — Reuse a pip-installed torch wheel (what this repo actually does)

If you already have Python + `pip install torch` somewhere (e.g. the
project's `rubik_training/venv`), that wheel **already contains** full
LibTorch headers and libraries under `site-packages/torch/{include,lib}` —
you don't need a separate download. This project's `CMakeLists.txt` uses
exactly this, pointed at `third_party/torch_cpu/` (see why below).

```powershell
python -m pip install --no-deps --target third_party\torch_cpu `
  --index-url https://download.pytorch.org/whl/cpu torch
```

**Why a *separate* CPU wheel instead of just using `rubik_training/venv`'s
existing `torch==...+cu121`?** That wheel's bundled `Caffe2Config.cmake`
unconditionally requires a full CUDA Toolkit to be *findable by CMake* to
configure at all — even if you only want a CPU build target. Rather than
install a multi-GB CUDA Toolkit just to satisfy that check, installing a
plain CPU wheel into an isolated folder sidesteps it entirely, at the cost
of CPU-only inference/training in the C++ tools. If you want CUDA-accelerated
training from C++, use Option A with a CUDA-platform download instead (and
do have the matching CUDA Toolkit installed).

Either way, the resulting `TorchConfig.cmake` under
`<torch root>/share/cmake/Torch/` is what `find_package(Torch)` needs.

## 3. Other prerequisites (Windows)

- **Visual Studio 2022**, with the **"Desktop development with C++"**
  workload installed (gives you `cl.exe`, `link.exe`). Check via:
  ```powershell
  & "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe" `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64
  ```
- **CMake ≥ 3.21** and **Ninja**. Easiest: `pip install cmake ninja` into any
  Python environment — gives you working binaries without a system installer.
- **OpenCV (C++)** — *not* `pip install opencv-python` (that's Python
  bindings only, no C++ headers/import libs). Get the official prebuilt
  Windows package: download `opencv-<version>-windows.exe` from
  <https://github.com/opencv/opencv/releases>, run it (it's a self-extracting
  archive, no installer/admin needed), and note where it extracted to (it
  creates an `opencv/build/` folder with `include/`, `x64/vc16/lib/`,
  `x64/vc16/bin/`). Point `RUBIK_OPENCV_DIR` at that `build/` folder — this
  project defaults to `third_party/opencv/build`.

## 4. Building this project

```powershell
cd rubik_training_cpp
.\configure.bat -DRUBIK_BUILD_TORCH_TOOLS=ON      # or .\configure_gpu.bat for GPU support
..\rubik_training\venv\Scripts\cmake.exe --build build
```

`configure.bat`/`configure_gpu.bat` run `vcvars64.bat` for you before
invoking CMake (needed so `cl.exe` is on `PATH`) — but that only applies
*inside* the batch script's own process; it doesn't carry over to your
shell. Plain `cmake` isn't on `PATH` at all on this machine (no system
install, only the venv's vendored copy), so the build step needs the full
path to it as shown above — a bare `cmake --build build` will fail with
"not recognized." `RUBIK_BUILD_TORCH_TOOLS=ON` is required for
`train_cube_pose_8pt`/`export_cube_pose_8pt`/`pseudo_label_real_cube_images`
(off by default so the non-ML tools build without needing Torch at all;
`configure_gpu.bat` turns it on automatically).

Executables and their runtime DLLs land in `build/`. Run any tool with
`--help` for its full flag list.

## 5. Preparing a dataset

This is the same pipeline `rubik_training/README_TRAINING.md` documents,
using the C++ tools instead of the Python scripts. All dataset folders (raw
CVAT sources + merged/YOLO outputs) live in `shared_datasets/`, a sibling of
`rubik_training/` and `rubik_training_cpp/` — both pipelines read/write the
same data there instead of each keeping its own copy. If you already have a
dataset prepared via the Python pipeline
(`../shared_datasets/pose_dataset_combined_cvat_8pt/`), skip straight to
[Section 7](#7-training).

**Note**: `merge_coco_keypoints_datasets` and `coco_keypoints_to_yolo_pose`
(both languages) clear their output directory before writing, specifically
so that regenerating a dataset after dropping a source can't leave stale
files from a previous run mixed into the current one — training reads these
directories directly, not a manifest, so leftover files silently end up in
the training set otherwise. (This is exactly what happened to
`pose_dataset_combined_cvat_8pt/` before it was caught and fixed — see
`README_TRAINING.md`.)

**5.1 — Record video(s) of the cube.** A few short clips (30-90s), varying
background, distance, angle, and lighting between clips. Slow, deliberate
motion — fast motion just produces blur that gets filtered out next.

**5.2 — Extract frames:**
```powershell
.\build\extract_video_frames.exe --video clip1.mp4 --video clip2.mp4 --fps 2 --max-frames 250
```
Samples ~2 frames/second, drops blurry frames (variance-of-Laplacian check),
writes JPEGs to `video_frames/`.

**5.3 — Thin near-duplicates (optional but recommended):**
```powershell
.\build\dedupe_similar_frames.exe --folder video_frames
```

**5.4 — Label in CVAT.** This step is manual and outside this project: create
a CVAT task from `video_frames/`, label class `cube` (bbox) + `cube_corners`
(8 keypoints, kp0-kp7, in a fixed order — front face TL/TR/BR/BL then back
face TL/TR/BR/BL, visibility 2=visible/1=occluded/0=missing). Export as
**COCO Keypoints 1.0**.

*Optional speed-up*: pre-label frames with an already-trained model so CVAT
opens with boxes/corners roughly in place (see `pseudo_label_real_cube_images`
in `README_CPP.md` — needs a TorchScript export, i.e. one produced by the
*Python* pipeline; see Section 6 for why the C++ pipeline can't produce one
of these itself yet).

**5.5 — Merge your labeled batches into one dataset:**
```powershell
.\build\merge_coco_keypoints_datasets.exe `
  --input-json path\to\export1.json --images-root path\to\images1 --dataset-tag batch1 `
  --input-json path\to\export2.json --images-root path\to\images2 --dataset-tag batch2 `
  --output-root combined_keypoints
```
Repeat `--input-json`/`--images-root`/`--dataset-tag` for as many labeled
batches as you have. Writes `combined_keypoints/images/all/` +
`combined_keypoints/annotations/merged_keypoints.json`.

**5.6 — Convert to the YOLO-pose training layout:**
```powershell
.\build\coco_keypoints_to_yolo_pose.exe `
  --coco-json combined_keypoints\annotations\merged_keypoints.json `
  --images-root combined_keypoints\images\all `
  --output-root pose_dataset_cpp
```
Writes `pose_dataset_cpp/images/{train,val}/`, `labels/{train,val}/`, and a
`data.yaml` — this is what `train_cube_pose_8pt` reads via `--data`.

*(Optional QA before training: `check_keypoint_label_consistency.exe` flags
mislabeled corner order; `visualize_keypoint_labels.exe` renders a contact
sheet to eyeball a batch.)*

## 6. Getting a model into the Flutter app — read this first

**If your goal is "a trained model running in `solve_my_cube`", train with
the existing Python pipeline, not this one — it already works end-to-end:**

```powershell
cd ..\rubik_training
.\venv\Scripts\python.exe train_cube_pose_8pt.py --prepare --device 0
# then, per colab_export_tflite.py: upload the resulting best.pt to a Google
# Colab notebook, export to TFLite, copy the .tflite into
# solve_my_cube/assets/models/cube_pose_8pt.tflite
```

That path exists precisely because Ultralytics' TFLite exporter needs
Linux/macOS (Colab), and the Python-side model is a standard `torch.jit.trace`
away from a TorchScript file TFLite conversion can consume.

**Why a bridge step is needed:** the Flutter app loads a TFLite model.
Producing that requires a TorchScript export of the trained weights. A
**traced Python `nn.Module`** exports to TorchScript easily (`torch.jit.trace`,
one line) — that's how the Python pipeline's `export_for_flutter()` works. A
**from-scratch `torch::nn::Module` trained directly in LibTorch C++** does not
have an equivalent easy path: LibTorch's C++ API doesn't expose a stable,
general "compile this arbitrary C++ module graph into a runnable TorchScript
file" tracer the way Python does. So `export_cube_pose_8pt` writes a
**LibTorch-native archive** (`torch::save`) — loadable only by `torch::load()`
back into a `rubik::YoloPose8pt` C++ instance in this codebase, not by
Python, `ultralytics`, or `colab_export_tflite.py` directly.

**This gap is now bridged** — see [`DEPLOY_TO_APP.md`](DEPLOY_TO_APP.md) for
the working, step-by-step path from a C++ checkpoint to a deployed `.tflite`
(bridge → TFLite export → copy into the app), plus its current verification
status (structurally verified; not yet proven bit-exact against a fully
trained checkpoint).

## 7. Training

Point `--data` at whichever dataset you actually have — the real one that
already exists after Section 5 (or from the Python pipeline) is
`shared_datasets/pose_dataset_combined_cvat_8pt`; `pose_dataset_cpp` in
Section 5's example command was just an illustrative name for a fresh
dataset, not something that exists on its own. Run from `rubik_training_cpp/`:

```powershell
.\build\train_cube_pose_8pt.exe `
  --data ..\shared_datasets\pose_dataset_combined_cvat_8pt `
  --epochs 240 --imgsz 640 --batch 8 --device 0 `
  --project runs_cpp\pose --name cube_corners_8pt_v1
```

- `--device 0` needs the GPU build (`configure_gpu.bat`, not plain
  `configure.bat` — see `README_CPP.md`'s "GPU build" section for the CUDA
  Toolkit install + the three CMake-level compatibility fixes it needed on
  this machine). Confirmed working end-to-end (clean rebuild, GPU
  correctly detected and used, loss decreases, graphs generate). Falls
  back to CPU with a warning if the LibTorch build doesn't have CUDA
  support (e.g. the plain CPU-only wheel from Option B) — use `--device
  cpu` explicitly in that case. Measured on this machine at the real
  training resolution (imgsz=640, batch=8): **~19s/epoch on GPU**, so 240
  epochs is roughly **75-80 minutes** (patience defaults to 60, so it may
  stop earlier if val loss plateaus). See `README_CPP.md`'s performance
  note for why the GPU/CPU gap isn't larger yet, and what the two
  optimizations already applied were.
- Trains from **random initialization** (no pretrained-weight loading — see
  `README_CPP.md`'s Phase 3 section for why), so expect it to need more
  epochs/data than the Python pipeline's COCO-pretrained fine-tune to reach
  comparable accuracy, if it gets there at all on a dataset this small.
- Checkpoints: `runs_cpp/pose/<name>/weights/{last,best}.pt`, saved every
  epoch (`best.pt` tracks lowest validation loss).
- Lower `--imgsz` (e.g. 160-320) and `--batch` for faster iteration while
  you're first getting a run working end-to-end; raise both for a real
  training run.

Once trained, `export_cube_pose_8pt --weights runs_cpp\pose\<name>\weights\best.pt
--output exported\model.pt` gives you the LibTorch archive described in
Section 6 — useful for further C++-side work, not for the app.

### Training graphs

After training finishes, the tool automatically evaluates the best checkpoint
on the val set and writes a set of plots into `runs_cpp/pose/<name>/`
(alongside `weights/`) — no separate step needed:

- `results.png` / `results_components.png` — loss curves over epochs.
- `confidence_hist.png` / `iou_hist.png` — distribution of the model's
  confidence and box accuracy on val images.
- `pr_f1_curve.png` — precision/recall/F1 vs. confidence threshold: whether
  the model's confidence score is actually trustworthy.
- `keypoint_error.png` — mean pixel error per corner (kp0-kp7), so a
  consistently-weak corner (e.g. an often-occluded back one) stands out.
- `val_predictions.jpg` — a contact sheet of ground truth (green) vs
  predicted (yellow) box+corners on a few val images, the fastest way to
  eyeball what's going wrong.

`graphs_readme.txt`, written into the same folder, has the same explanations
for later reference. See `README_CPP.md`'s Phase 3 section for how each
number is computed.
