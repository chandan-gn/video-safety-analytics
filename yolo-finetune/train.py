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

