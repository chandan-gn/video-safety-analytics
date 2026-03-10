"""
Fine-tune YOLOv8m on SH17 with stratified splitting and per-class weighted loss.

Changes from train.py:
  - No reliance on pre-existing train_files.txt / val_files.txt / test_files.txt
  - Stratified 80/20 train/val split per class (multi-label iterative stratification)
  - Per-class inverse-frequency weighted BCE loss during training
  - Per-class AP50 printed at validation

Requires:
  pip install scikit-multilearn

Run:
    python train_strat.py
    python train_strat.py --config config.yaml
"""
import argparse
import tempfile
from pathlib import Path

import numpy as np
import yaml
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.yaml")
args = parser.parse_args()

with open(ROOT / args.config) as f:
    cfg = yaml.safe_load(f)

IMAGES_DIR = ROOT / cfg["images_dir"]
LABELS_DIR = ROOT / cfg["labels_dir"]
WEIGHTS    = ROOT / cfg["weights"]
YAML       = ROOT / cfg["dataset_yaml"]

EPOCHS    = cfg["epochs"]
IMG_SIZE  = cfg["img_size"]
BATCH     = cfg["batch"]
DEVICE    = cfg["device"]
PROJECT   = ROOT / "runs"
RUN_NAME  = cfg["run_name"]
SEED      = cfg["seed"]
VAL_SPLIT = cfg["val_split"]

with open(YAML) as f:
    _ds = yaml.safe_load(f)
CLASS_NAMES = list(_ds["names"].values())
NUM_CLASSES = len(CLASS_NAMES)

# ---------------------------------------------------------------------------
# Step 1 — Build label matrix
# ---------------------------------------------------------------------------
def build_label_matrix():
    """Return (image_names list, binary label matrix [N, C])."""
    exts = {".jpg", ".jpeg", ".png"}
    image_names = []
    label_matrix = []

    for f in sorted(IMAGES_DIR.iterdir()):
        if f.suffix.lower() not in exts:
            continue
        lf = LABELS_DIR / (f.stem + ".txt")
        if not lf.exists():
            continue
        classes_present = set()
        for line in lf.read_text().splitlines():
            parts = line.strip().split()
            if parts:
                classes_present.add(int(parts[0]))
        image_names.append(f.name)
        label_matrix.append([1 if c in classes_present else 0 for c in range(NUM_CLASSES)])

    if not image_names:
        raise FileNotFoundError(
            f"No image/label pairs found.\n"
            f"  images_dir : {IMAGES_DIR}\n"
            f"  labels_dir : {LABELS_DIR}"
        )

    return image_names, np.array(label_matrix)


# ---------------------------------------------------------------------------
# Step 2 — Stratified split
# ---------------------------------------------------------------------------
def stratified_split(image_names, label_matrix):
    """Multi-label stratified 80/20 split using iterative stratification."""
    try:
        from skmultilearn.model_selection import iterative_train_test_split
    except ImportError:
        raise ImportError("Run: pip install scikit-multilearn")

    np.random.seed(SEED)
    X = np.arange(len(image_names)).reshape(-1, 1)
    X_train, y_train, X_val, y_val = iterative_train_test_split(
        X, label_matrix, test_size=VAL_SPLIT
    )
    return X_train.flatten(), y_train, X_val.flatten(), y_val


def print_split_report(train_idx, y_train, val_idx, y_val):
    print(f"\n[split] Train: {len(train_idx)}  Val: {len(val_idx)}")
    print(f"\n  {'Class':<16} {'Train':>6} {'Val':>6} {'Train%':>8} {'Val%':>8}")
    print(f"  {'-'*16} {'-'*6} {'-'*6} {'-'*8} {'-'*8}")
    for c in range(NUM_CLASSES):
        t = int(y_train[:, c].sum())
        v = int(y_val[:, c].sum())
        total = t + v
        if total == 0:
            continue
        print(f"  {CLASS_NAMES[c]:<16} {t:>6} {v:>6} {t/total*100:>7.1f}% {v/total*100:>7.1f}%")


# ---------------------------------------------------------------------------
# Step 3 — Per-class loss weights
# ---------------------------------------------------------------------------
def compute_class_weights(y_train, max_ratio=10.0):
    """Inverse-frequency weights derived from training set label matrix."""
    counts = y_train.sum(axis=0).astype(float).clip(1)
    weights = counts.sum() / (NUM_CLASSES * counts)   # inverse frequency
    weights = weights / weights.min()                  # min weight = 1.0
    weights = weights.clip(1.0, max_ratio)

    print(f"\n  {'Class':<16} {'Images':>7} {'Weight':>8}")
    print(f"  {'-'*16} {'-'*7} {'-'*8}")
    for c in range(NUM_CLASSES):
        print(f"  {CLASS_NAMES[c]:<16} {int(counts[c]):>7} {weights[c]:>8.2f}")

    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Step 4 — Write temp files for ultralytics
# ---------------------------------------------------------------------------
def write_temp_split(image_names, train_idx, val_idx):
    """Write temporary txt files — avoids touching existing split files."""
    def _write(indices):
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, dir=ROOT
        )
        for i in indices:
            tmp.write(f"./data/images/{image_names[i]}\n")
        tmp.close()
        return Path(tmp.name)

    return _write(train_idx), _write(val_idx)


def write_temp_yaml(train_file, val_file):
    """Write a temporary dataset yaml pointing to the stratified split files."""
    with open(YAML) as f:
        ds = yaml.safe_load(f)
    ds["train"] = str(train_file)
    ds["val"]   = str(val_file)
    ds["test"]  = ""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, dir=ROOT
    )
    yaml.dump(ds, tmp)
    tmp.close()
    return Path(tmp.name)


def cleanup(*paths):
    for p in paths:
        try:
            Path(p).unlink()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Step 5 — Train with per-class weighted loss
# ---------------------------------------------------------------------------
def train(class_weights, dataset_yaml):
    from ultralytics import YOLO
   # from ultralytics.yolo.v8.detect import DetectionTrainer
    from ultralytics.models.yolo.detect import DetectionTrainer


    class BalancedDetectionTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            import types
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            _w = class_weights  # capture for closure

            # criterion is lazily built on first forward pass via init_criterion()
            # override it here so the weighted BCE is injected at that point
            def custom_init_criterion(self_model):
                from ultralytics.utils.loss import v8DetectionLoss
                loss = v8DetectionLoss(self_model)
                loss.bce = nn.BCEWithLogitsLoss(
                    pos_weight=_w.to(loss.device), reduction="none"
                )
                print(f"[trainer] Per-class weighted BCE injected — "
                      f"min={_w.min():.2f} max={_w.max():.2f}")
                return loss

            model.init_criterion = types.MethodType(custom_init_criterion, model)
            return model

    print("\n[train] Loading weights:", WEIGHTS)
    model = YOLO(str(WEIGHTS))
    model.train(
        data=str(dataset_yaml),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        project=str(PROJECT),
        name=RUN_NAME,
        exist_ok=True,
        seed=SEED,
        val=True,
        save=True,
        plots=True,
        verbose=True,
        trainer=BalancedDetectionTrainer,
    )


# ---------------------------------------------------------------------------
# Step 6 — Validate and report per-class AP50
# ---------------------------------------------------------------------------
def validate(weights_path, dataset_yaml):
    from ultralytics import YOLO

    print(f"\n[validate] Evaluating {weights_path}")
    model = YOLO(str(weights_path))
    return model.val(
        data=str(dataset_yaml),
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        split="val",
        plots=True,
        verbose=True,
    )


def print_metrics(metrics):
    print("\n" + "=" * 50)
    print("  VALIDATION METRICS (best.pt)")
    print("=" * 50)
    try:
        box = metrics.box
        print(f"  mAP50       : {box.map50:.4f}")
        print(f"  mAP50-95    : {box.map:.4f}")
        print(f"  Precision   : {box.mp:.4f}")
        print(f"  Recall      : {box.mr:.4f}")
        print()
        print(f"  {'Class':<16} {'AP50':>8}")
        print(f"  {'-'*16} {'-'*8}")
        for i, name in enumerate(CLASS_NAMES):
            if i < len(box.ap50):
                print(f"  {name:<16} {box.ap50[i]:>8.4f}")
    except Exception:
        print(metrics.results_dict)
    print("=" * 50 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 50)
    print("  SH17 YOLOv8m — Stratified + Weighted Loss")
    print("=" * 50)

    print("\n[step 1] Scanning image/label pairs ...")
    image_names, label_matrix = build_label_matrix()
    print(f"         Found {len(image_names)} matched pairs across {NUM_CLASSES} classes")

    print("\n[step 2] Stratified 80/20 split per class ...")
    train_idx, y_train, val_idx, y_val = stratified_split(image_names, label_matrix)
    print_split_report(train_idx, y_train, val_idx, y_val)

    print("\n[step 3] Computing per-class loss weights ...")
    class_weights = compute_class_weights(y_train)

    print("\n[step 4] Writing temporary split files ...")
    train_file, val_file = write_temp_split(image_names, train_idx, val_idx)
    dataset_yaml = write_temp_yaml(train_file, val_file)
    print(f"         train → {train_file.name}")
    print(f"         val   → {val_file.name}")

    try:
        print("\n[step 5] Training ...")
        train(class_weights, dataset_yaml)

        best_weights = PROJECT / RUN_NAME / "weights" / "best.pt"
        if best_weights.exists():
            print("\n[step 6] Final validation ...")
            metrics = validate(best_weights, dataset_yaml)
            print_metrics(metrics)
            print(f"Best weights: {best_weights}")
        else:
            print(f"[warn] best.pt not found at {best_weights}")
    finally:
        cleanup(train_file, val_file, dataset_yaml)
        print("[cleanup] Temporary split files removed.")


main()

