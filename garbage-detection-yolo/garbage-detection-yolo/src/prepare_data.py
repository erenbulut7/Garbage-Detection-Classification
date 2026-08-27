"""
prepare_data.py

Pools all images/labels from the original Roboflow train/valid/test split and
creates a new, class-stratified 70/15/15 split.

Why this script exists:
The original dataset split (as downloaded from Roboflow) was found to have a
severe class imbalance issue -- e.g. the GLASS class had zero examples in the
original test set. This script fixes that by pooling all data and creating a
stratified split where every class is proportionally represented in each subset.

Usage:
    python prepare_data.py --input_dir /path/to/original_dataset \
                            --output_dir /path/to/resplit_dataset \
                            --train_ratio 0.7 --val_ratio 0.15 --seed 42
"""

import argparse
import os
import random
import shutil
from collections import defaultdict


def pool_dataset(input_dir, pool_img_dir, pool_lbl_dir):
    """Copy all images/labels from train/valid/test into a single pool."""
    os.makedirs(pool_img_dir, exist_ok=True)
    os.makedirs(pool_lbl_dir, exist_ok=True)

    for split in ["train", "valid", "test"]:
        img_src = os.path.join(input_dir, split, "images")
        lbl_src = os.path.join(input_dir, split, "labels")
        if not os.path.isdir(img_src):
            print(f"Warning: {img_src} not found, skipping.")
            continue
        for f in os.listdir(img_src):
            shutil.copy(os.path.join(img_src, f), os.path.join(pool_img_dir, f))
        for f in os.listdir(lbl_src):
            shutil.copy(os.path.join(lbl_src, f), os.path.join(pool_lbl_dir, f))

    n_img = len(os.listdir(pool_img_dir))
    n_lbl = len(os.listdir(pool_lbl_dir))
    print(f"Pooled {n_img} images and {n_lbl} label files.")
    return n_img, n_lbl


def stratified_split(pool_lbl_dir, train_ratio, val_ratio, seed):
    """
    Assign each label file (and its corresponding image) to train/val/test,
    processing rarest classes first so every class gets fair representation
    across all three subsets.
    """
    random.seed(seed)

    label_files = [f for f in os.listdir(pool_lbl_dir) if f.endswith(".txt")]

    class_to_imgs = defaultdict(list)
    for lf in label_files:
        with open(os.path.join(pool_lbl_dir, lf)) as f:
            classes = set(int(line.split()[0]) for line in f if line.strip())
        for c in classes:
            class_to_imgs[c].append(lf)

    assigned = {}
    train_set, val_set, test_set = set(), set(), set()

    # Process rarest class first to ensure minority classes are split fairly
    for c in sorted(class_to_imgs, key=lambda c: len(class_to_imgs[c])):
        imgs = [i for i in class_to_imgs[c] if i not in assigned]
        random.shuffle(imgs)
        n = len(imgs)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        for i, img in enumerate(imgs):
            if i < n_train:
                train_set.add(img)
                assigned[img] = "train"
            elif i < n_train + n_val:
                val_set.add(img)
                assigned[img] = "val"
            else:
                test_set.add(img)
                assigned[img] = "test"

    print(f"Split sizes -> train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)}")
    return train_set, val_set, test_set


def write_split(pool_img_dir, pool_lbl_dir, output_dir, split_name, label_files, img_ext=".jpg"):
    img_out = os.path.join(output_dir, split_name, "images")
    lbl_out = os.path.join(output_dir, split_name, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    for lf in label_files:
        img_name = lf.replace(".txt", img_ext)
        src_img = os.path.join(pool_img_dir, img_name)
        if not os.path.exists(src_img):
            # fall back to other common extensions
            for alt_ext in [".jpg", ".jpeg", ".png"]:
                alt = os.path.join(pool_img_dir, lf.replace(".txt", alt_ext))
                if os.path.exists(alt):
                    src_img = alt
                    img_name = lf.replace(".txt", alt_ext)
                    break
        shutil.copy(src_img, os.path.join(img_out, img_name))
        shutil.copy(os.path.join(pool_lbl_dir, lf), os.path.join(lbl_out, lf))


def main():
    parser = argparse.ArgumentParser(description="Pool and re-split the garbage classification dataset.")
    parser.add_argument("--input_dir", required=True, help="Path to original dataset (with train/valid/test subfolders)")
    parser.add_argument("--output_dir", required=True, help="Path to write the resplit dataset")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pool_img_dir = os.path.join(args.output_dir, "_pool", "images")
    pool_lbl_dir = os.path.join(args.output_dir, "_pool", "labels")

    pool_dataset(args.input_dir, pool_img_dir, pool_lbl_dir)
    train_set, val_set, test_set = stratified_split(pool_lbl_dir, args.train_ratio, args.val_ratio, args.seed)

    write_split(pool_img_dir, pool_lbl_dir, args.output_dir, "train", train_set)
    write_split(pool_img_dir, pool_lbl_dir, args.output_dir, "valid", val_set)
    write_split(pool_img_dir, pool_lbl_dir, args.output_dir, "test", test_set)

    # clean up the temporary pool
    shutil.rmtree(os.path.join(args.output_dir, "_pool"))

    print(f"\nDone. Resplit dataset written to: {args.output_dir}")
    print("Remember to update data.yaml 'path:' to point to this new directory.")


if __name__ == "__main__":
    main()
