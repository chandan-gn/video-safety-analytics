"""
Fine-tune YOLOv8m on the SH17 PPE Detection Dataset.

Steps:
  0. YOLO prefers full path to the images, so, update to absolute path of the images before training
  1. Regenerate 80/20 train/val split from images that have matching labels.
  2. Train for the specified number of epochs.
  3. Validate best weights and print final metrics.

Run from the project root:
    python train.py
    will run with config.yaml file, to override the default config.yaml and to use the custom run
    python train.py --config config_2version.yaml
"""
import os
import random
import json
from pathlib import Path
import yaml

# ---------------------------------------------------------------------------
# Config — edit config.yaml
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.resolve()

with open(ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)

IMAGES_DIR  = ROOT / cfg["images_dir"]
LABELS_DIR  = ROOT / cfg["labels_dir"]
WEIGHTS     = ROOT / cfg["weights"]
YAML        = ROOT / cfg["dataset_yaml"]
TRAIN_FILE  = ROOT / cfg["train_file"]
VAL_FILE    = ROOT / cfg["val_file"]

EPOCHS      = cfg["epochs"]
IMG_SIZE    = cfg["img_size"]
BATCH       = cfg["batch"]
DEVICE      = cfg["device"]
PROJECT     = ROOT / "runs"
RUN_NAME    = cfg["run_name"]
SEED        = cfg["seed"]
VAL_SPLIT   = cfg["val_split"]
# ---------------------------------------------------------------------------
def make_split():
    """Scan disk for image/label pairs and write 80/20 split list files."""
    exts = {".jpg", ".jpeg", ".png"}
    images = sorted(
        f.name for f in IMAGES_DIR.iterdir()
        if f.suffix.lower() in exts
        and (LABELS_DIR / (f.stem + ".txt")).exists()
    )
    if not images:
        raise FileNotFoundError(
            f"No image/label pairs found.\n"
            f"  images_dir: {IMAGES_DIR}\n"
            f"  labels_dir: {LABELS_DIR}"
        )

    random.seed(SEED)
    random.shuffle(images)
    split = int((1 - VAL_SPLIT) * len(images))
    train_imgs, val_imgs = images[:split], images[split:]

    def write(path, img_list):
        with open(path, "w") as f:
            for img in img_list:
                f.write(f"./data/images/{img}\n")

    write(TRAIN_FILE, train_imgs)
    write(VAL_FILE,   val_imgs)
    print(f"[split] Train: {len(train_imgs)}  Val: {len(val_imgs)}  "
          f"(total matched pairs: {len(images)})")
    return len(train_imgs), len(val_imgs)

def train():
    from ultralytics import YOLO

    print("\n[train] Loading weights:", WEIGHTS)
    model = YOLO(str(WEIGHTS))

    results = model.train(
        data=str(YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        project=str(PROJECT),
        name=RUN_NAME,
        exist_ok=True,          # resume / overwrite same run folder
        seed=SEED,
        val=True,               # run validation each epoch
        save=True,              # save best.pt and last.pt
        plots=True,             # save training curves
        verbose=True,
    )
    return results

def validate(weights_path):
    from ultralytics import YOLO

    print(f"\n[validate] Evaluating {weights_path}")
    model = YOLO(str(weights_path))
    metrics = model.val(
        data=str(YAML),
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        split="val",
        plots=True,
        verbose=True,
    )
    return metrics


def print_metrics(metrics):
    """Print a concise metric summary."""
    print("\n" + "=" * 50)
    print("  VALIDATION METRICS (best.pt)")
    print("=" * 50)
    try:
        box = metrics.box
        print(f"  mAP50       : {box.map50:.4f}")
        print(f"  mAP50-95    : {box.map:.4f}")
        print(f"  Precision   : {box.mp:.4f}")
        print(f"  Recall      : {box.mr:.4f}")
    except Exception:
        # Fallback: print raw results dict
        print(metrics.results_dict)
    print("=" * 50 + "\n")


def main():
    print("=" * 50)
    print("  SH17 YOLOv8m Fine-Tuning")
    print("=" * 50)

    # Step 1: Build split files
    print("\n[step 1] Generating train/val split ...")
    make_split()

    # Step 2: Train
    print("\n[step 2] Starting training ...")
    train()

    # Step 3: Validate best weights
    best_weights = PROJECT / RUN_NAME / "weights" / "best.pt"
    if best_weights.exists():
        print("\n[step 3] Running final validation ...")
        metrics = validate(best_weights)
        print_metrics(metrics)
        print(f"Best weights saved at: {best_weights}")
    else:
        print(f"[warn] best.pt not found at {best_weights} — check training output.")


main()
