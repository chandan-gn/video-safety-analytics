"""
Fine-tune YOLOv8m on the SH17 PPE Detection Dataset.

Steps:
  0. YOLO prefers full path to the images, so, update to absolute path of the images before training
  1. Regenerate 80/20 train/val split from images that have matching labels.
  2. Train for the specified number of epochs.
  3. Validate best weights and print final metrics.

Run from the project root:
    python train.py
"""
import os
import random
import json
from pathlib import path

# ---------------------------------------------------------------------------
# Config — edit these as needed
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).parent.resolve()
IMAGES_DIR  = ROOT / "data" / "images"
LABELS_DIR  = ROOT / "data" / "labels"
WEIGHTS     = ROOT / "model" / "yolo8m.pt"
YAML        = ROOT / "sh17_kaggle.yaml"
TRAIN_FILE  = ROOT / "train_files.txt"
VAL_FILE    = ROOT / "val_files.txt"

EPOCHS      = 1
IMG_SIZE    = 320
BATCH       = 8          # lower batch for CPU; raise to 16 if you have GPU RAM
DEVICE      = "cpu"      # "cuda:0" if GPU is available
PROJECT     = ROOT / "runs"
RUN_NAME    = "sh17_yolo8m"
SEED        = 42
VAL_SPLIT   = 0.2
# ---------------------------------------------------------------------------

