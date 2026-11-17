#!/usr/bin/env python3
"""
SKU-110K → YOLO Format Converter
==================================

Converts SKU-110K CSV annotations into YOLO-standard directory structure
with normalized bounding box labels.

Usage:
    python scripts/convert_sku110k.py

Input:  data/sku110k/annotations/annotations_{train,val,test}.csv
        data/sku110k/images/
Output: data/sku110k_yolo/images/{train,val,test}/
        data/sku110k_yolo/labels/{train,val,test}/
"""

import os
import sys
import pandas as pd
import shutil
from pathlib import Path


def setup_yolo_directories(base_dir: str, split: str) -> tuple[str, str]:
    """Creates YOLO-standard images and labels directories.

    Args:
        base_dir: Root output directory for YOLO dataset.
        split: Dataset split name ('train', 'val', or 'test').

    Returns:
        Tuple of (image_directory_path, label_directory_path).
    """
    img_dir = os.path.join(base_dir, 'images', split)
    lbl_dir = os.path.join(base_dir, 'labels', split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    return img_dir, lbl_dir


def convert_sku110k_to_yolo(
    csv_path: str,
    source_img_dir: str,
    base_output_dir: str,
    split: str,
) -> None:
    """Converts a single SKU-110K annotation CSV to YOLO label format.

    Each row in the CSV represents one bounding box. The function groups
    by image, copies images to the YOLO directory, and writes one .txt
    label file per image with normalized [class cx cy w h] format.

    Args:
        csv_path: Path to the SKU-110K annotations CSV file.
        source_img_dir: Directory containing the source images.
        base_output_dir: Root directory for the YOLO-formatted output.
        split: Dataset split name ('train', 'val', or 'test').

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    print(f"Processing {split} split...")
    img_dest_dir, lbl_dest_dir = setup_yolo_directories(base_output_dir, split)

    # SKU-110K CSVs do not have headers
    columns = [
        'image_name', 'x1', 'y1', 'x2', 'y2',
        'class', 'image_width', 'image_height',
    ]
    try:
        df = pd.read_csv(csv_path, names=columns)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}. Make sure dataset is extracted.")
        return

    grouped = df.groupby('image_name')
    total = len(grouped)
    skipped = 0

    for idx, (image_name, group) in enumerate(grouped):
        if (idx + 1) % 500 == 0 or idx == 0:
            print(f"  [{split}] {idx + 1}/{total} images processed...")

        # 1. Copy image to YOLO images directory
        src_img_path = os.path.join(source_img_dir, image_name)
        dst_img_path = os.path.join(img_dest_dir, image_name)

        if os.path.exists(src_img_path):
            if not os.path.exists(dst_img_path):
                shutil.copy2(src_img_path, dst_img_path)
        else:
            skipped += 1
            continue

        # 2. Convert and write labels
        txt_filename = os.path.splitext(image_name)[0] + '.txt'
        txt_filepath = os.path.join(lbl_dest_dir, txt_filename)

        with open(txt_filepath, 'w') as f:
            for _, row in group.iterrows():
                img_w = row['image_width']
                img_h = row['image_height']

                # Normalize coordinates to [0, 1]
                x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
                box_w = (x2 - x1) / img_w
                box_h = (y2 - y1) / img_h
                center_x = (x1 + (x2 - x1) / 2) / img_w
                center_y = (y1 + (y2 - y1) / 2) / img_h

                # Clamp to [0, 1] to handle annotation edge cases
                center_x = max(0.0, min(1.0, center_x))
                center_y = max(0.0, min(1.0, center_y))
                box_w = max(0.0, min(1.0, box_w))
                box_h = max(0.0, min(1.0, box_h))

                # class_id is always 0 (single-class: "product")
                f.write(f"0 {center_x:.6f} {center_y:.6f} {box_w:.6f} {box_h:.6f}\n")

    print(f"Finished {split}: {total - skipped} images converted, {skipped} skipped.")
    print(f"  Labels → {lbl_dest_dir}")
    print(f"  Images → {img_dest_dir}")


if __name__ == "__main__":
    # Resolve paths relative to project root (one level up from scripts/)
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    SKU_BASE = os.path.join('data', 'sku110k')
    YOLO_BASE = os.path.join('data', 'sku110k_yolo')

    splits = {
        'train': os.path.join(SKU_BASE, 'annotations', 'annotations_train.csv'),
        'val':   os.path.join(SKU_BASE, 'annotations', 'annotations_val.csv'),
        'test':  os.path.join(SKU_BASE, 'annotations', 'annotations_test.csv'),
    }

    print(f"Project root: {project_root}")
    print(f"Output dir:   {os.path.abspath(YOLO_BASE)}")
    print("=" * 60)

    for split_name, csv_file in splits.items():
        convert_sku110k_to_yolo(
            csv_file,
            os.path.join(SKU_BASE, 'images'),
            YOLO_BASE,
            split_name,
        )

    print("\n✓ Conversion complete. You can now train with:")
    print(f"  yolo detect train data=configs/sku110k.yaml model=yolov8m.pt epochs=50 imgsz=640")
