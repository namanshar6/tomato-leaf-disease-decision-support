"""
data_prep.py
------------
Filters the tomato-only classes out of a downloaded PlantVillage dataset
and organizes them into:

    data/train/<class_name>/*.jpg
    data/val/<class_name>/*.jpg

Usage:
    1. Download the PlantVillage dataset from Kaggle (see README.md for the
       exact commands). After downloading + unzipping, you should have a
       folder somewhere with subfolders per class, e.g.:
           raw_data/Tomato___Bacterial_spot/*.jpg
           raw_data/Tomato___healthy/*.jpg
           raw_data/Pepper__bell___healthy/*.jpg   <-- not tomato, ignored
           ...

    2. Run:
           python data_prep.py --source raw_data --dest data --val_split 0.2

This will find every class folder whose name contains "tomato" (case
insensitive), and split its images 80/20 (or whatever you choose) into
data/train/<class> and data/val/<class>.
"""

import argparse
import os
import random
import shutil
from pathlib import Path


def find_tomato_class_dirs(source_dir: Path):
    """Recursively find directories whose name contains 'tomato'."""
    matches = []
    for root, dirs, files in os.walk(source_dir):
        for d in dirs:
            if "tomato" in d.lower():
                full = Path(root) / d
                # Only treat as a class folder if it directly contains images
                imgs = [f for f in os.listdir(full)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))]
                if imgs:
                    matches.append(full)
    return matches


def clean_class_name(name: str) -> str:
    # Normalize names like "Tomato___Bacterial_spot" -> "Tomato_Bacterial_spot"
    name = name.replace("___", "_").replace(" ", "_")
    return name


def split_and_copy(class_dir: Path, dest_root: Path, val_split: float, seed: int):
    class_name = clean_class_name(class_dir.name)
    images = [f for f in os.listdir(class_dir)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.Random(seed).shuffle(images)

    n_val = max(1, int(len(images) * val_split))
    val_imgs = images[:n_val]
    train_imgs = images[n_val:]

    train_dir = dest_root / "train" / class_name
    val_dir = dest_root / "val" / class_name
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    for f in train_imgs:
        shutil.copy2(class_dir / f, train_dir / f)
    for f in val_imgs:
        shutil.copy2(class_dir / f, val_dir / f)

    print(f"[{class_name}] total={len(images)} train={len(train_imgs)} val={len(val_imgs)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to raw downloaded PlantVillage data")
    parser.add_argument("--dest", default="data", help="Destination folder (will contain train/ and val/)")
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_dir = Path(args.source)
    dest_root = Path(args.dest)

    if dest_root.exists():
        print(f"Destination '{dest_root}' already exists — new files will be added alongside it.")
    dest_root.mkdir(parents=True, exist_ok=True)

    tomato_dirs = find_tomato_class_dirs(source_dir)
    if not tomato_dirs:
        raise SystemExit(
            f"No tomato class folders found under '{source_dir}'. "
            "Check that you pointed --source at the extracted dataset root."
        )

    print(f"Found {len(tomato_dirs)} tomato class folders:")
    for d in tomato_dirs:
        print("  -", d.name)
    print()

    for class_dir in tomato_dirs:
        split_and_copy(class_dir, dest_root, args.val_split, args.seed)

    print("\nDone. Your dataset is ready at:")
    print(f"  {dest_root / 'train'}")
    print(f"  {dest_root / 'val'}")


if __name__ == "__main__":
    main()
