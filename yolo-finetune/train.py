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

