#!/usr/bin/env python3
# prepare_cat_dataset.py
"""
Download the Kaggle cat-breeds dataset, unpack everything, and copy/resize
all images into `cropped_cats/<label>/` folders ready for training.
"""

import os
import shutil
import zipfile
import tarfile
from pathlib import Path

import kagglehub
import cv2     # sudo apt install python3-opencv  (-headless is fine)
               # pip install opencv-python-headless if not already present

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION ─ change these if you like
DATASET_ID        = "ma7555/cat-breeds-dataset"
TARGET_BASE       = Path("cropped_cats")     # where classifier expects images
IMG_SIZE          = (128, 128)               # set to None to keep original size
VALID_EXTENSIONS  = {".jpg", ".jpeg", ".png"}
# ──────────────────────────────────────────────────────────────────────────────


def is_archive(path: Path) -> bool:
    return path.suffix in {".zip", ".gz", ".tgz", ".tar"}


def extract_archive(archive_path: Path, dest_dir: Path) -> None:
    """Extract .zip or .tar[.gz] into dest_dir (safely)."""
    print(f"Extracting {archive_path.name} …")
    dest_dir.mkdir(parents=True, exist_ok=True)

    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dest_dir)
    elif archive_path.suffix in {".gz", ".tgz", ".tar"}:
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(dest_dir)
    else:
        print(f"Unknown archive type: {archive_path}")
    print("   - done.")


def collect_images(dataset_root: Path, target_root: Path) -> None:
    """Walk through dataset_root, copy/resize images into target_root/label/."""
    count = 0
    for file in dataset_root.rglob("*"):
        # Skip directories & non-image files
        if not file.is_file() or file.suffix.lower() not in VALID_EXTENSIONS:
            continue

        # The parent folder name becomes the label (normalize spaces → _)
        label = file.parent.name.strip().lower().replace(" ", "_")
        dest_dir = target_root / label
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest_file = dest_dir / file.name

        # Read, (optionally) resize, and save
        img = cv2.imread(str(file))
        if img is None:
            print(f"Could not read {file}, skipping.")
            continue
        if IMG_SIZE is not None:
            img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dest_file), img)
        count += 1

    print(f"Copied {count} images into {target_root}")


def main() -> None:
    # 1. Download dataset
    print("⬇Downloading dataset …")
    dataset_path_str = kagglehub.dataset_download(DATASET_ID)
    dataset_path = Path(dataset_path_str)
    print("Dataset downloaded to:", dataset_path)

    # 2. Extract any archives first-level down (if present)
    for item in dataset_path.iterdir():
        if item.is_file() and is_archive(item):
            extract_archive(item, item.parent / item.stem)  # extract beside

    # 3. Collect images recursively into target folder
    TARGET_BASE.mkdir(exist_ok=True)
    collect_images(dataset_root=dataset_path, target_root=TARGET_BASE)


if __name__ == "__main__":
    main()
