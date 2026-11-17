"""
End-to-End Compliance Pipeline Module
======================================

Orchestrates the entire multi-stage retail audit workflow:
1. Product Detection (YOLOv8 or Ground Truth Bounding Boxes for evaluation).
2. Feature Extraction (ResNet-50 2048-d unit-normalized embeddings).
3. SKU Recognition & Open-Set Rejection (SKUMatcher vs. Faiss GalleryIndex).
4. Spatial Row Clustering (ShelfRowClusterer: top-to-bottom, left-to-right).
5. Planogram Sequence Alignment (PlanogramAligner: edit distance).
6. Compliance Scoring & Annotated Visual Output Generation.
"""

from typing import List, Dict, Tuple, Any, Optional, Union
import time
from pathlib import Path
import numpy as np
from PIL import Image

from src.recognizer import FeatureExtractor, SKUMatcher
from src.gallery import GalleryIndex
from src.planogram import ShelfRowClusterer, PlanogramAligner
from src.utils import load_image_rgb, extract_crops, draw_planogram_audit


class PlanogramCompliancePipeline:
    """End-to-end shelf auditing engine.

    Attributes:
        extractor (FeatureExtractor): Visual feature extraction network.
        gallery (GalleryIndex): Vector index of reference SKUs.
        matcher (SKUMatcher): Nearest-neighbor search + open-set classifier.
        clusterer (ShelfRowClusterer): 2D bbox row clustering algorithm.
        aligner (PlanogramAligner): Sequence alignment and compliance auditor.
    """

    def __init__(
        self,
        gallery: GalleryIndex,
        extractor: Optional[FeatureExtractor] = None,
        match_threshold: float = 0.65,
        reject_threshold: float = 0.40,
        vertical_tolerance: float = 0.50
    ) -> None:
        """Initializes all sub-modules for the compliance pipeline.

        Args:
            gallery: Populated GalleryIndex instance.
            extractor: Optional FeatureExtractor (if None, instantiates default ResNet50).
            match_threshold: Confirmed SKU match similarity cutoff [0, 1].
            reject_threshold: Similarity cutoff below which items are flagged UNKNOWN.
            vertical_tolerance: Relative vertical margin multiplier for row grouping.
        """
        assert gallery is not None, "GalleryIndex cannot be None."
        self.gallery = gallery
        self.extractor = extractor or FeatureExtractor()
        self.matcher = SKUMatcher(match_threshold=match_threshold, reject_threshold=reject_threshold)
        self.clusterer = ShelfRowClusterer(vertical_tolerance=vertical_tolerance)
        self.aligner = PlanogramAligner()

    def audit_shelf(
        self,
        shelf_image: Union[str, Path, Image.Image, np.ndarray],
        expected_planogram_rows: List[List[str]],
        detected_boxes: Optional[List[List[float]]] = None,
        is_normalized_boxes: bool = False
    ) -> Dict[str, Any]:
        """Audits a shelf image against an expected reference planogram.

        Args:
            shelf_image: Raw shelf image (filepath, PIL Image, or ndarray).
            expected_planogram_rows: Expected SKU layout (list of rows, each having list of SKU IDs).
            detected_boxes: Optional list of bounding boxes [x1, y1, x2, y2].
                If None, you can plug in YOLOv8 detector outputs.
            is_normalized_boxes: Whether detected_boxes coordinates are normalized to [0, 1].

        Returns:
            Dictionary containing:
                - 'overall_compliance_rate': Percentage compliance [0.0, 100.0].
                - 'total_summary': Aggregated counts of CORRECT, MISPLACED, MISSING, UNKNOWN.
                - 'row_alignments': Detailed slot-by-slot alignment for each shelf row.
                - 'annotated_image': PIL Image with color-coded audit visualization.
                - 'inference_time_seconds': Total wall-clock time taken.
        """
        start_time = time.perf_counter()
        img = load_image_rgb(shelf_image)

        if not detected_boxes:
            # No products detected on shelf
            elapsed = time.perf_counter() - start_time
            return {
                "overall_compliance_rate": 0.0,
                "total_summary": {"CORRECT": 0, "MISPLACED": 0, "MISSING": sum(len(r) for r in expected_planogram_rows), "UNKNOWN": 0},
                "row_alignments": [],
                "annotated_image": img,
                "inference_time_seconds": round(elapsed, 4)
            }

        # Filter valid bounding boxes (drop NaNs / Infs from corrupted datasets)
        valid_boxes = []
        for b in detected_boxes:
            if len(b) == 4 and not any(np.isnan(b)) and not any(np.isinf(b)):
                valid_boxes.append(b)

        if not valid_boxes:
            elapsed = time.perf_counter() - start_time
            return {
                "overall_compliance_rate": 0.0,
                "total_summary": {"CORRECT": 0, "MISPLACED": 0, "MISSING": sum(len(r) for r in expected_planogram_rows), "UNKNOWN": 0},
                "row_alignments": [],
                "annotated_image": img,
                "inference_time_seconds": round(elapsed, 4)
            }

        # 1. Crop each product facing from the shelf image
        crops = extract_crops(img, valid_boxes, is_normalized=is_normalized_boxes)

        # 2. Extract visual embeddings in batch
        embeddings = self.extractor.extract_batch(crops, batch_size=64)

        # 3. Recognize SKUs and evaluate open-set rejection threshold
        detections: List[Dict[str, Any]] = []
        for idx, (box, emb) in enumerate(zip(valid_boxes, embeddings)):
            match_res = self.matcher.match(emb, self.gallery)
            detections.append({
                "detection_id": idx,
                "bbox": box,
                "sku_id": match_res["sku_id"],
                "confidence": match_res["confidence"],
                "recognition_status": match_res["status"],
                "top_candidates": match_res["top_candidates"]
            })

        # 4. Spatially group detections into horizontal shelf rows (top-to-bottom, left-to-right)
        detected_rows = self.clusterer.cluster_detections(detections)

        # 5. Sequence alignment against expected planogram
        audit_report = self.aligner.align_shelf(expected_planogram_rows, detected_rows)

        # 6. Generate annotated visualization
        all_aligned_slots: List[Dict[str, Any]] = []
        for r in audit_report["row_alignments"]:
            all_aligned_slots.extend(r["alignment"])

        annotated_img = draw_planogram_audit(img, all_aligned_slots)

        elapsed = time.perf_counter() - start_time
        audit_report["annotated_image"] = annotated_img
        audit_report["inference_time_seconds"] = round(elapsed, 4)

        return audit_report
