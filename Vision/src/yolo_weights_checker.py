"""
Inspect a YOLO weights file and print its task, architecture, and class names.
Supports Ultralytics YOLO models (.pt) — detection, classification, segmentation, pose, OBB.

Set WEIGHTS_PATH below and run the script directly, or pass the path as a CLI argument.
"""

import sys
from pathlib import Path

# ── Set your weights path here ────────────────────────────────────────────────
WEIGHTS_PATH = r"C:/Users/ignac/Escritorio/DualVision/Utilities/Vision/workbench/weights/occupancy_v1.pt"
# ──────────────────────────────────────────────────────────────────────────────


def check_weights(weights_path: str) -> None:
    path = Path(weights_path)
    if not path.exists():
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)
    if path.suffix != ".pt":
        print(f"[WARN] Expected a .pt file, got: {path.suffix}")

    try:
        import torch
    except ImportError:
        print("[ERROR] PyTorch is not installed. Run: pip install torch")
        sys.exit(1)

    # Validate ZIP magic before torch.load to give a clear error on truncated files
    with open(path, "rb") as _f:
        header = _f.read(4)
        if header != b"PK\x03\x04":
            print("[ERROR] Not a valid PyTorch ZIP archive (bad magic bytes). File may be corrupted.")
            sys.exit(1)
        _f.seek(-22, 2)
        eocd = _f.read(4)
        if eocd != b"PK\x05\x06":
            print("[ERROR] ZIP central directory not found at end of file.")
            print("        The file is likely truncated — re-download or re-copy it.")
            sys.exit(1)

    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)

    print(f"\n{'='*60}")
    print(f"  Weights file : {path.name}")
    print(f"{'='*60}")

    model_obj = ckpt.get("model") if isinstance(ckpt, dict) else ckpt
    _print_from_model_obj(model_obj)

    if isinstance(ckpt, dict):
        _print_raw_metadata(ckpt)

    print()


def _print_from_model_obj(model_obj) -> None:
    if model_obj is None:
        return

    task = getattr(model_obj, "task", None)
    yaml_cfg = getattr(model_obj, "yaml", None) or {}
    yaml_task = yaml_cfg.get("task", None)
    detected_task = task or yaml_task or _infer_task(model_obj)
    print(f"\n  Task         : {detected_task or 'unknown'}")

    arch = yaml_cfg.get("yaml_file") or yaml_cfg.get("backbone") or type(model_obj).__name__
    scale = yaml_cfg.get("scale", "")
    print(f"  Architecture : {arch}{f'  (scale: {scale})' if scale else ''}")

    ch = yaml_cfg.get("ch", None)
    if ch:
        print(f"  Input ch     : {ch}")

    names = getattr(model_obj, "names", None) or yaml_cfg.get("names", None)
    if names:
        if isinstance(names, dict):
            num_classes = len(names)
            class_list = [f"{k}: {v}" for k, v in sorted(names.items())]
        else:
            num_classes = len(names)
            class_list = [f"{i}: {n}" for i, n in enumerate(names)]
        print(f"  Num classes  : {num_classes}")
        print("  Classes      :")
        for c in class_list:
            print(f"    {c}")
    else:
        nc = yaml_cfg.get("nc", None)
        if nc:
            print(f"  Num classes  : {nc}  (no class names stored)")

    kpt_shape = yaml_cfg.get("kpt_shape", None)
    if kpt_shape:
        print(f"  Keypoints    : {kpt_shape}")


def _print_raw_metadata(ckpt: dict) -> None:
    interesting = {
        "epoch": "Last epoch",
        "best_fitness": "Best fitness",
        "train_args": "Train args",
        "date": "Date trained",
        "version": "Ultralytics version",
    }
    extras = {label: ckpt[key] for key, label in interesting.items() if key in ckpt and ckpt[key] is not None}

    if extras:
        print("\n  -- Checkpoint metadata --")
        for label, val in extras.items():
            if label == "Train args" and hasattr(val, "__dict__"):
                val = vars(val)
            if isinstance(val, dict):
                sub_keys = ["imgsz", "batch", "epochs", "optimizer", "lr0", "data", "project"]
                sub = {k: val[k] for k in sub_keys if k in val}
                if sub:
                    print(f"  {label:<20}: {sub}")
            else:
                print(f"  {label:<20}: {val}")


def _infer_task(model_obj) -> str:
    try:
        head = model_obj.model[-1]
        head_name = type(head).__name__
        for key, task in {"Detect": "detection", "Segment": "segmentation",
                          "Classify": "classification", "Pose": "pose", "OBB": "obb"}.items():
            if key in head_name:
                return task
    except Exception:
        pass
    return None


if __name__ == "__main__":
    # CLI argument takes priority over the hardcoded variable
    path = sys.argv[1] if len(sys.argv) > 1 else WEIGHTS_PATH
    if not path:
        print("Usage: python yolo_weights_checker.py <path/to/weights.pt>")
        print("       or set WEIGHTS_PATH at the top of the script.")
        sys.exit(1)
    check_weights(path)
