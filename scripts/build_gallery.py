#!/usr/bin/env python3
"""
Pre-compute and Persist GroZi Gallery Feature Index
===================================================

Scans all studio reference images in data/grozi/Training/, extracts
2048-dimensional L2-normalized ResNet-50 embeddings, and persists:
  1. Faiss vector index: outputs/gallery_index/gallery.index
  2. Metadata catalog:   outputs/gallery_index/gallery_metadata.json

Usage:
    python scripts/build_gallery.py [--batch-size 64] [--gallery-dir data/grozi/Training]
"""

import argparse
import time
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.recognizer import FeatureExtractor
from src.gallery import GalleryIndex


def parse_args():
    parser = argparse.ArgumentParser(description="Build and save GroZi Faiss gallery index.")
    parser.add_argument(
        "--gallery-dir",
        type=str,
        default="data/grozi/Training",
        help="Root directory containing GroZi reference images."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/gallery_index",
        help="Directory to save gallery.index and gallery_metadata.json."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Inference batch size for feature extraction."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Execution device ('cuda' or 'cpu')."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    gallery_path = Path(args.gallery_dir)
    output_path = Path(args.output_dir)

    print("=" * 65)
    print("🛒 RETAIL PLANOGRAM COMPLIANCE — GALLERY BUILDER")
    print("=" * 65)
    print(f"Gallery Source : {gallery_path.resolve()}")
    print(f"Destination    : {output_path.resolve()}")
    print(f"Batch Size     : {args.batch_size}")
    print(f"Compute Device : {args.device.upper()}")
    print("=" * 65)

    if not gallery_path.exists():
        raise FileNotFoundError(f"Source gallery directory does not exist: {gallery_path}")

    start_t = time.perf_counter()

    # 1. Initialize Feature Extractor (ResNet50)
    print("\n[1/3] Loading ResNet-50 Feature Extractor...")
    extractor = FeatureExtractor(backbone="resnet50", device=args.device)

    # 2. Build Faiss Gallery
    print("\n[2/3] Extracting features and populating Faiss IndexFlatIP...")
    gallery = GalleryIndex(embedding_dim=extractor.embedding_dim)
    total_indexed = gallery.build_from_directory(
        gallery_dir=gallery_path,
        extractor=extractor,
        batch_size=args.batch_size
    )

    # 3. Persist to disk
    print("\n[3/3] Saving index and metadata to disk...")
    gallery.save(output_path)

    elapsed = time.perf_counter() - start_t
    print("\n" + "=" * 65)
    print(f"✓ DONE: Successfully indexed {total_indexed} SKUs in {elapsed:.2f}s ({elapsed/60:.2f} min).")
    print(f"✓ Index file   : {output_path / 'gallery.index'}")
    print(f"✓ Metadata file: {output_path / 'gallery_metadata.json'}")
    print("=" * 65)


if __name__ == "__main__":
    main()
