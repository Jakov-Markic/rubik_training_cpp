"""Bridges a LibTorch C++-trained cube-pose checkpoint into a real ultralytics.YOLO
model, so the existing colab_export_tflite.py path can turn it into a TFLite file for
the Flutter app.

Why this is needed: train_cube_pose_8pt.exe's checkpoints are LibTorch-native archives
(torch::save format) or, via export_cube_pose_8pt.exe --pickle-out, a Python-loadable
pickle of {name: tensor} using *this codebase's own* module names (e.g. "stem.conv.weight",
"box_head0.0.conv.weight") -- not ultralytics' names (e.g. "model.0.conv.weight",
"model.22.cv2.0.0.conv.weight"). Neither is something `ultralytics.YOLO()` can load
directly. This script copies every tensor across using a verified name-correspondence
table (see rubik_training_cpp/README_CPP.md's "Deployment bridge" section for how it
was derived and verified -- shapes were diffed layer-by-layer against a real trained
ultralytics checkpoint before this table was written, and --verify below re-checks it
numerically against this codebase's own decode on the same images).

Usage (run in Colab, or locally in rubik_training/venv -- both have ultralytics+torch):

    python bridge_to_ultralytics.py \
        --cpp-state cpp_state.pt \
        --reference best.pt \
        --output bridged.pt

--cpp-state: produced by
    export_cube_pose_8pt.exe --weights <your_cpp_checkpoint> --pickle-out cpp_state.pt
--reference: any real ultralytics checkpoint with the right architecture (nc=1,
    kpt_shape=[8,3]) -- e.g. rubik_training/runs/pose/cube_corners_8pt_cvat_v4/weights/best.pt.
    Only its architecture is used; its weights are entirely overwritten by --cpp-state's.
--output: where to write the bridged, ultralytics-loadable checkpoint.

Then export it exactly like a normally-trained checkpoint (see colab_export_tflite.py):

    from ultralytics import YOLO
    m = YOLO("bridged.pt")
    m.export(format="tflite", imgsz=320, nms=False, half=False)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


# (this codebase's module name, ultralytics' module name) for every module whose
# parameters/buffers need copying. "dfl" is a fixed, non-trainable buffer (softmax bin
# weights, arange(reg_max)) that's identical by construction on both sides -- mapped
# anyway (harmless, already-equal values) just so strict=True load_state_dict below
# doesn't need a special case for it.
MODULE_MAP: list[tuple[str, str]] = [
    ("dfl", "model.22.dfl"),
    # Backbone
    ("stem", "model.0"),
    ("down1", "model.1"),
    ("c2f_p2", "model.2"),
    ("down2", "model.3"),
    ("c2f_p3", "model.4"),
    ("down3", "model.5"),
    ("c2f_p4", "model.6"),
    ("down4", "model.7"),
    ("c2f_p5", "model.8"),
    ("sppf", "model.9"),
    # Neck (PAN-FPN)
    ("neck_c2f1", "model.12"),
    ("neck_c2f2", "model.15"),
    ("neck_down1", "model.16"),
    ("neck_c2f3", "model.18"),
    ("neck_down2", "model.19"),
    ("neck_c2f4", "model.21"),
    # Head: box branch (cv2), cls branch (cv3), keypoint branch (cv4), one per FPN level.
    ("box_head0", "model.22.cv2.0"),
    ("box_head1", "model.22.cv2.1"),
    ("box_head2", "model.22.cv2.2"),
    ("cls_head0", "model.22.cv3.0"),
    ("cls_head1", "model.22.cv3.1"),
    ("cls_head2", "model.22.cv3.2"),
    ("kpt_head0", "model.22.cv4.0"),
    ("kpt_head1", "model.22.cv4.1"),
    ("kpt_head2", "model.22.cv4.2"),
]


def build_remapped_state(cpp_state: dict, reference_state: dict) -> dict:
    remapped = {}
    used_cpp_keys = set()

    for cpp_prefix, ultra_prefix in MODULE_MAP:
        cpp_dot = cpp_prefix + "."
        ultra_dot = ultra_prefix + "."
        matched = [k for k in cpp_state if k.startswith(cpp_dot)]
        if not matched:
            raise RuntimeError(f"No cpp_state keys found with prefix '{cpp_dot}' -- checkpoint mismatch?")
        for cpp_key in matched:
            suffix = cpp_key[len(cpp_dot):]
            ultra_key = ultra_dot + suffix
            if ultra_key not in reference_state:
                raise RuntimeError(
                    f"Mapped key '{ultra_key}' (from '{cpp_key}') doesn't exist in the reference "
                    "checkpoint -- architecture mismatch. Check --reference is a real nc=1/kpt_shape=[8,3] "
                    "ultralytics pose checkpoint."
                )
            if cpp_state[cpp_key].shape != reference_state[ultra_key].shape:
                raise RuntimeError(
                    f"Shape mismatch: '{cpp_key}' {tuple(cpp_state[cpp_key].shape)} vs "
                    f"'{ultra_key}' {tuple(reference_state[ultra_key].shape)}"
                )
            remapped[ultra_key] = cpp_state[cpp_key]
            used_cpp_keys.add(cpp_key)

    unused = set(cpp_state.keys()) - used_cpp_keys
    if unused:
        print(f"WARNING: {len(unused)} cpp_state keys were never used by the mapping: {sorted(unused)[:10]} ...")

    return remapped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cpp-state", type=Path, required=True, help="Pickle from export_cube_pose_8pt --pickle-out")
    parser.add_argument("--reference", type=Path, required=True, help="A real ultralytics pose checkpoint (architecture template)")
    parser.add_argument("--output", type=Path, required=True, help="Where to write the bridged checkpoint")
    parser.add_argument(
        "--verify-images",
        type=Path,
        nargs="*",
        default=[],
        help="Optional: run the bridged model on these images and print the same "
        "'SAMPLE i conf=... box=... kp0=...' format export_cube_pose_8pt/train_cube_pose_8pt "
        "--eval-only prints, for a direct numeric diff against this codebase's own decode.",
    )
    parser.add_argument("--verify-imgsz", type=int, default=320, help="Must match what --verify-images were captured/trained at.")
    args = parser.parse_args()

    from ultralytics import YOLO

    print(f"Loading C++ state: {args.cpp_state}")
    cpp_state = torch.load(args.cpp_state, map_location="cpu", weights_only=False)
    print(f"  {len(cpp_state)} entries")

    print(f"Loading reference architecture: {args.reference}")
    ref_model = YOLO(str(args.reference))
    reference_state = ref_model.model.state_dict()
    print(f"  {len(reference_state)} entries")

    print("Building remapped state dict...")
    remapped = build_remapped_state(cpp_state, reference_state)
    print(f"  mapped {len(remapped)} / {len(reference_state)} reference entries")

    missing = set(reference_state.keys()) - set(remapped.keys())
    if missing:
        raise RuntimeError(f"{len(missing)} reference entries were never mapped: {sorted(missing)[:10]} ...")

    ref_model.model.load_state_dict(remapped, strict=True)
    ref_model.model.eval()
    print("load_state_dict: OK (strict=True, every tensor accounted for)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ref_model.save(str(args.output))
    print(f"Wrote bridged checkpoint: {args.output}")

    if args.verify_images:
        verify(ref_model, args.verify_images, args.verify_imgsz)


def verify(model, image_paths: list[Path], imgsz: int) -> None:
    """Mirrors export_cube_pose_8pt/train_cube_pose_8pt --eval-only's decode exactly:
    letterbox to imgsz (scale-to-fit, gray114, centered pad), raw forward pass (the
    Pose head's eval-mode output is already DFL/keypoint-decoded, matching
    PoseSingleBestWrapper's input format -- see train_cube_pose_8pt.py), pick the
    single highest-confidence anchor, print normalized box+keypoints the same way."""
    import cv2
    import numpy as np

    print(f"\nVerifying on {len(image_paths)} image(s) at imgsz={imgsz}:")
    net = model.model
    net.eval()

    for i, img_path in enumerate(image_paths):
        bgr = cv2.imread(str(img_path))
        h, w = bgr.shape[:2]
        long_side = max(h, w)
        scale = 1.0 if long_side <= imgsz else imgsz / long_side
        cw, ch = max(1, round(w * scale)), max(1, round(h * scale))
        content = bgr if (cw, ch) == (w, h) else cv2.resize(bgr, (cw, ch), interpolation=cv2.INTER_LINEAR)
        pad_x, pad_y = (imgsz - cw) // 2, (imgsz - ch) // 2
        canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + ch, pad_x:pad_x + cw] = content

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).contiguous()

        with torch.no_grad():
            out = net(tensor)
            if isinstance(out, tuple):
                out = out[0]
            out = out[0]  # (4 + nc + 3*nkpt, num_anchors)

        boxes = out[0:4]
        conf = out[4]
        kpts = out[5:5 + 3 * 8]

        best_idx = int(torch.argmax(conf))
        best_conf = float(conf[best_idx])
        cx, cy, bw, bh = (float(boxes[j, best_idx]) for j in range(4))
        left, top = (cx - bw / 2) / imgsz, (cy - bh / 2) / imgsz
        right, bottom = (cx + bw / 2) / imgsz, (cy + bh / 2) / imgsz

        kp = kpts[:, best_idx].view(8, 3)
        kp_str = " ".join(
            f"kp{k}={float(kp[k,0])/imgsz:.6f},{float(kp[k,1])/imgsz:.6f},{float(kp[k,2]):.6f}" for k in range(8)
        )
        print(f"SAMPLE {i} conf={best_conf:.6f} box={left:.6f},{top:.6f},{right:.6f},{bottom:.6f} {kp_str}")


if __name__ == "__main__":
    main()
