# Getting a trained checkpoint into the Flutter app

## TL;DR

You trained with `train_cube_pose_8pt.exe` (C++). To get that model into the
app:

1. **Bridge** it into a real ultralytics checkpoint (commands below) — this
   now works, verified structurally correct but not yet bit-exact-proven
   (see [Verification status](#verification-status)).
2. **Export** that bridged checkpoint to TFLite, on Colab or WSL2 (Ultralytics'
   exporter needs Linux/macOS, not Windows).
3. **Copy** the resulting `.tflite` into `solve_my_cube/assets/models/`.

Full commands for all three steps are below. If you'd rather skip the
bridge entirely, [Option A](#option-a--use-the-python-pipeline-instead)
covers using the already-working Python-trained checkpoint instead.

## Why a bridge step is needed at all

`train_cube_pose_8pt.exe`'s checkpoints are **LibTorch-native archives**
(`torch::save` format) — not ultralytics checkpoints, not TorchScript
files. `ultralytics.YOLO()`, `torch.jit.load()`, and every TFLite
conversion tool have no idea what to do with one directly. Colab and WSL2
both exist to solve a *different* problem (running Ultralytics' TFLite
exporter somewhere other than native Windows) — neither helps until the
checkpoint is already a real ultralytics model in memory.

## Step 1 — Bridge the checkpoint

Two commands: one in C++ (exports your weights as a plain, Python-readable
file), one in Python (copies them into a real ultralytics model). Run both
locally — no Colab needed for this step, `rubik_training/venv` already has
everything required.

```powershell
# From rubik_training_cpp\, with your C++ checkpoint already trained:
.\build\export_cube_pose_8pt.exe `
  --weights runs_cpp\pose\cube_corners_8pt_v1\weights\best.pt `
  --pickle-out bridge_out\cpp_state.pt
```

```powershell
# From rubik_training\:
.\venv\Scripts\python.exe ..\rubik_training_cpp\tools\bridge_to_ultralytics.py `
  --cpp-state ..\rubik_training_cpp\bridge_out\cpp_state.pt `
  --reference runs\pose\cube_corners_8pt_cvat_v4\weights\best.pt `
  --output ..\rubik_training_cpp\bridge_out\bridged.pt
```

`--reference` just supplies the correct *architecture* (any real,
already-trained ultralytics checkpoint with `nc=1, kpt_shape=[8,3]` works —
`cube_corners_8pt_cvat_v4`'s is fine) — its weights are entirely
overwritten by `--cpp-state`'s. You'll get a `bridged.pt` that
`ultralytics.YOLO()` can load like any other checkpoint.

**Add `--verify-images <path1> <path2> ...` and `--verify-imgsz <N>`** to
print the bridged model's raw predictions on specific images, in the same
format `train_cube_pose_8pt.exe --eval-only <checkpoint>` prints for the
original C++ model — useful for sanity-checking a specific bridge before
trusting it (see [Verification status](#verification-status)).

## Step 2 — Export the bridged checkpoint to TFLite

*(This is the same step `rubik_training/colab_export_tflite.py` already
documents for a normally Python-trained checkpoint — `bridged.pt` behaves
identically to any other ultralytics checkpoint from here on.)*

Ultralytics' TFLite exporter needs Linux or macOS. Two ways to get one:

### Colab (cloud, zero local setup)

1. Go to <https://colab.research.google.com>, new notebook.
2. ```python
   !pip install -q ultralytics
   from google.colab import files
   uploaded = files.upload()  # choose bridged.pt
   ```
3. ```python
   from ultralytics import YOLO
   model = YOLO("bridged.pt")
   exported = model.export(format="tflite", imgsz=320, nms=False, half=False)
   ```
4. ```python
   import shutil
   shutil.copy(exported, "cube_pose_8pt.tflite")
   files.download("cube_pose_8pt.tflite")
   ```

Pros: free, nothing to install. Cons: uploads your checkpoint to Google's
servers, needs internet, manual upload/download every time you export a
new checkpoint.

### WSL2 (local, one-time setup, nothing leaves your machine)

WSL2 is a real Ubuntu Linux environment running locally on the same
Windows machine — no cloud, no upload, and it reads your Windows files
directly.

**One-time setup:**

```powershell
# In an elevated (Administrator) PowerShell:
wsl --install
# Reboot if prompted, then open the new "Ubuntu" app from the Start menu
# and create a Linux username/password when asked.
```

```bash
# Inside the Ubuntu terminal:
sudo apt update && sudo apt install -y python3-pip python3-venv
python3 -m venv ~/tflite-export
source ~/tflite-export/bin/activate
pip install ultralytics
```

**Running an export** (repeat this part each time — your Windows `C:`
drive is mounted at `/mnt/c/` inside WSL, so no file copying needed):

```bash
source ~/tflite-export/bin/activate
yolo export \
  model=/mnt/c/Users/User/Desktop/SolveMeCube/rubik_training_cpp/bridge_out/bridged.pt \
  format=tflite imgsz=320 nms=False half=False
```

The resulting `.tflite` lands right there via `/mnt/c/` — nothing to
download or transfer.

### Colab vs. WSL2

| | Colab | WSL2 |
|---|---|---|
| Setup | None | ~1-2GB install, one-time |
| Your checkpoint leaves your machine? | Yes (uploaded to Google) | No |
| Needs internet to run an export | Yes | No (after setup) |
| Repeat exports | Manual upload/download each time | Just rerun one command |

If you'll be iterating (retraining, re-bridging, re-exporting) more than
once, WSL2 pays for its setup cost quickly. Colab is the lower-commitment
way to try the whole flow once.

## Step 3 — Get the `.tflite` into the app

1. Copy it to `solve_my_cube/assets/models/cube_pose_8pt.tflite`,
   replacing the existing one.
2. Confirm the export's `imgsz` (320 above) matches
   `RubikDetector.inputSize` in
   `solve_my_cube/lib/services/detector_service.dart` (also 320 by
   default) — they're commented pointing at each other specifically so a
   mismatch is easy to catch.
3. ```powershell
   cd solve_my_cube
   flutter pub get
   flutter run
   ```

## Verification status

The bridge's name-correspondence table (`rubik_training_cpp/tools/bridge_to_ultralytics.py`'s
`MODULE_MAP`) wasn't guessed — it was derived by dumping real ultralytics
parameter shapes from an actual trained checkpoint and diffing them
layer-by-layer against this C++ architecture. Result: the backbone+neck's
135 parameters matched **positionally with zero shape mismatches**; the
head matched by explicit group (box/cls/kpt branches × 3 FPN levels).
`load_state_dict(strict=True)` succeeds — all 397 tensors accounted for,
zero shape errors.

**Update (2026-08-24), re-tested on a real converged checkpoint:** `--verify-images`
was re-run against `runs_cpp/pose/cube_corners_8pt_v1` (236 epochs, healthy
convergence — not the earlier 3-epoch smoke test) on 5 val images, compared
directly against `train_cube_pose_8pt.exe --eval-only` on the same
checkpoint/images. Result is a mix of good and concerning:

- **Box position and all 8 keypoints matched closely** (within roughly 1-4%
  of frame) on every sample, and the same anchor/region was clearly selected
  each time — so the earlier "different anchor wins on a barely-converged
  model" theory is confirmed *not* the explanation, because that noise is
  gone on a real model and the geometry still lines up. This part of the
  bridge looks solid.
- **Confidence did not match, and not as small noise — as a large, one-directional
  gap on every single sample**: bridged confidence was lower than the C++
  model's own value every time (0.041 vs 0.366, 0.746 vs 0.967, 0.630 vs
  0.936, 0.631 vs 0.952, 0.804 vs 0.966 — roughly 20-90% relative drop,
  never the other direction). A same-magnitude floating-point kernel
  difference would be expected to scatter in both directions and shrink on
  a confident, converged model; this is neither. That points to a real
  remaining discrepancy specifically in how confidence/objectness is
  computed — somewhere between the C++ head's forward pass and what
  `ultralytics.YOLO`'s pose head (which the bridged checkpoint is loaded
  into) actually computes for that channel — not in the bridged weights
  themselves, since the box/keypoint branches (different weights, same
  bridge mechanism) came through fine.

**Bottom line: do not deploy this bridge yet.** The keypoint geometry
matching well is a good sign the weight-copying itself is correct, but a
confidence gap this large and this consistent means the exported model's
detection-confidence gating (`RubikDetector.minDetectionScore` and
`keypointVisibleThreshold` in the Flutter app) would behave differently
than intended — likely under-detecting real cubes, since every confidence
value came through *lower*. Root-causing this means comparing the C++
confidence-head implementation (`model/include/rubik/yolo_pose_8pt.hpp`)
against ultralytics' pose head's actual objectness computation
op-by-op — not yet done. Until then, use [Option A](#option-a--use-the-python-pipeline-instead).

## Option A — use the Python pipeline instead

If you'd rather sidestep the bridge entirely: the Python training pipeline
(`rubik_training/train_cube_pose_8pt.py`) produces a real ultralytics
checkpoint directly, with a longer production track record.

- **Retrain**: `cd rubik_training`, then
  `.\venv\Scripts\python.exe train_cube_pose_8pt.py --prepare --device 0`.
- **Reuse the existing one**:
  `rubik_training/runs/pose/cube_corners_8pt_cvat_v4/weights/best.pt` is
  already trained and sitting there (this is what the app currently ships
  from — see `README_TRAINING.md` for a caveat about a dataset
  contamination issue found and fixed since it was trained).

  **Watch out**: that same `weights/` folder also contains
  `best.torchscript`, `cube_pose_8pt.torchscript`, and
  `cube_pose_8pt.ptl` — already-exported TorchScript/mobile files sitting
  right next to the real checkpoint with easy-to-confuse names. Only
  `best.pt` (6.5MB) is the raw checkpoint Step 2 above expects; feeding it
  one of the others produces `"best.pt is a TorchScript archive, not an
  Ultralytics PyTorch checkpoint"` or similar.

Either way, once you have a real ultralytics `.pt` (from here, or from the
bridge above), it's the same Step 2 / Step 3 above.
