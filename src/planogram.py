"""
F4 - Planogram Alignment & Compliance Module
=============================================

This module provides:
1. ShelfRowClusterer: Groups 2D detected bounding boxes into discrete horizontal shelf rows
   (top-to-bottom, left-to-right) using adaptive vertical gap/overlap clustering.
2. PlanogramParser: Standardizes planar graph definitions (from GP-180 and GroZi)
   into row-by-row SKU sequences.
3. PlanogramAligner: Compares detected SKU sequences against reference planogram sequences
   using dynamic programming sequence alignment (Levenshtein edit distance) to classify:
   - CORRECT: Expected SKU matches detected SKU.
   - MISPLACED: Product present in wrong slot or unexpected position.
   - MISSING: Product on reference planogram absent on physical shelf.
   - UNKNOWN: Detected product could not be identified by open-set recognizer.

Designed for robust enterprise SQA validation with strict assertions, edge-case checks,
and high execution speed.
"""

from typing import List, Dict, Tuple, Any, Optional, Union
import json
import re
from pathlib import Path
import numpy as np


class ShelfRowClusterer:
    """Clusters 2D bounding boxes into sorted horizontal shelf rows.

    Handles camera tilts, perspective distortions, and varying product heights
    by sorting boxes vertically and grouping them into the same shelf row when their
    vertical center distances or vertical IoU overlaps fall within an adaptive threshold.

    Attributes:
        vertical_tolerance (float): Maximum normalized vertical distance ratio
            relative to average box height to merge into the same row (default 0.50).
    """

    def __init__(self, vertical_tolerance: float = 0.50) -> None:
        """Initializes the clusterer.

        Args:
            vertical_tolerance: Relative vertical margin multiplier [0.1, 1.0].
        """
        assert 0.05 <= vertical_tolerance <= 1.5, "vertical_tolerance must be between 0.05 and 1.5"
        self.vertical_tolerance = vertical_tolerance

    def cluster_detections(
        self,
        detections: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """Groups detections into shelf rows ordered from top to bottom, left to right.

        Args:
            detections: List of detection dictionaries. Each must contain:
                - 'bbox': [x1, y1, x2, y2] (pixel or normalized coordinates)
                Optional keys: 'sku_id', 'confidence', 'status', etc.

        Returns:
            List of shelf rows (from top row to bottom row).
            Each row is a list of detections sorted left-to-right (by x1).

        Raises:
            ValueError: If any detection lacks a valid 4-element 'bbox'.
        """
        if not detections:
            return []

        # Validate bounding boxes
        processed = []
        for idx, det in enumerate(detections):
            if "bbox" not in det or len(det["bbox"]) != 4:
                raise ValueError(f"Detection at index {idx} missing valid 4-element 'bbox': {det}")
            x1, y1, x2, y2 = det["bbox"]
            assert x2 >= x1 and y2 >= y1, f"Malformed bbox coordinates at index {idx}: {det['bbox']}"
            y_center = (y1 + y2) / 2.0
            height = max(1.0, y2 - y1)
            processed.append({
                **det,
                "_y_center": y_center,
                "_height": height,
                "_x1": x1
            })

        # Sort all items primarily by vertical center
        processed.sort(key=lambda d: d["_y_center"])

        # Adaptive row grouping
        rows: List[List[Dict[str, Any]]] = []
        row_y_centers: List[float] = []
        row_avg_heights: List[float] = []

        for item in processed:
            y_c = item["_y_center"]
            h = item["_height"]

            assigned_row = -1
            for r_idx, (r_yc, r_h) in enumerate(zip(row_y_centers, row_avg_heights)):
                threshold = max(h, r_h) * self.vertical_tolerance
                if abs(y_c - r_yc) <= threshold:
                    assigned_row = r_idx
                    break

            if assigned_row != -1:
                # Add to existing row and update running vertical average
                rows[assigned_row].append(item)
                n = len(rows[assigned_row])
                row_y_centers[assigned_row] = ((n - 1) * row_y_centers[assigned_row] + y_c) / n
                row_avg_heights[assigned_row] = ((n - 1) * row_avg_heights[assigned_row] + h) / n
            else:
                # Start a new shelf row
                rows.append([item])
                row_y_centers.append(y_c)
                row_avg_heights.append(h)

        # Sort rows top-to-bottom by average y-center
        combined = sorted(zip(row_y_centers, rows), key=lambda pair: pair[0])
        sorted_rows: List[List[Dict[str, Any]]] = []

        for _, r_items in combined:
            # Sort each row horizontally left-to-right by x1
            r_items.sort(key=lambda d: d["_x1"])
            # Clean up internal sorting helper keys
            clean_items = []
            for d in r_items:
                clean_d = {k: v for k, v in d.items() if not k.startswith("_")}
                clean_items.append(clean_d)
            sorted_rows.append(clean_items)

        return sorted_rows


class PlanogramParser:
    """Parses planar graph planograms (GP-180 & GroZi format) into shelf row sequences."""

    @staticmethod
    def normalize_sku_id(img_path: str) -> str:
        """Standardizes raw image paths into uniform SKU identifiers.

        Example:
            'Food/Tea/67.jpg' -> 'Food/Tea/67'
            'X:/Grocery stores/.../TrainingFood/Snacks/25.jpg' -> 'Food/Snacks/25'
        """
        clean = img_path.replace("\\", "/")
        # Strip extension
        clean = Path(clean).with_suffix("").as_posix()
        # Look for category starting points like Food/, HealthCare/, etc.
        match = re.search(r"(Background|Beers|CleaningProducts|Clothes|Food|HealthCare|HouseProducts|Kids|Pets|Plants|Spirits|Wines)/.*", clean)
        if match:
            return match.group(0)
        return Path(clean).name

    @classmethod
    def parse_gp180_json(cls, json_path: Union[str, Path]) -> List[List[str]]:
        """Parses a GP-180 planogram JSON containing 'objects' and 'graph' connectivity.

        Follows horizontal 'e' (east) and 'w' (west) edges for rows, and
        's' (south) / 'n' (north) edges for moving between shelves.

        Args:
            json_path: Path to GP-180 JSON file.

        Returns:
            List of shelf rows from top to bottom, where each row is a list of SKU IDs (left-to-right).
        """
        p = Path(json_path)
        if not p.exists():
            raise FileNotFoundError(f"Planogram file does not exist: {p}")

        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        objects = data.get("objects", data.get("oggetti", []))
        graph = data.get("graph", data.get("grafo", []))

        assert len(objects) > 0, f"Planogram {p} contains no objects."
        assert len(graph) > 0, f"Planogram {p} contains no graph nodes."

        # Map object index to normalized SKU ID
        sku_lookup: Dict[int, str] = {}
        for idx, obj in enumerate(objects):
            # In GroZi graph, 'ogg' is 1-indexed or 0-indexed reference
            sku_lookup[idx] = cls.normalize_sku_id(obj.get("img_path", f"SKU_{idx}"))

        # Find top-left nodes (nodes that have no North and no West neighbors)
        node_visited = set()
        top_left_nodes: List[int] = []

        for idx, node in enumerate(graph):
            is_top = (node.get("n", -1) == -1 and node.get("nw", -1) == -1 and node.get("ne", -1) == -1)
            is_left = (node.get("w", -1) == -1 and node.get("nw", -1) == -1 and node.get("sw", -1) == -1)
            if is_top and is_left:
                top_left_nodes.append(idx)

        # Fallback: traverse row-by-row starting from nodes without west neighbors
        left_nodes = [idx for idx, n in enumerate(graph) if n.get("w", -1) == -1]
        if not left_nodes:
            # Graph might be single linear or unordered, return all objects as a single row
            return [[cls.normalize_sku_id(obj["img_path"]) for obj in objects]]

        rows: List[List[str]] = []
        current_left = top_left_nodes[0] if top_left_nodes else left_nodes[0]
        visited_rows = set()

        while current_left != -1 and current_left not in visited_rows:
            visited_rows.add(current_left)
            row_skus: List[str] = []
            curr = current_left

            while curr != -1 and curr not in node_visited:
                node_visited.add(curr)
                node_info = graph[curr]
                # ogg field connects node to object
                ogg_idx = node_info.get("ogg", curr)
                # Handle 1-based indexing if ogg > 0 and len matches
                if ogg_idx > 0 and ogg_idx <= len(objects) and 0 not in [g.get("ogg") for g in graph]:
                    target_obj_idx = ogg_idx - 1
                else:
                    target_obj_idx = min(ogg_idx, len(objects) - 1)

                sku_id = sku_lookup.get(target_obj_idx, f"SKU_{target_obj_idx}")
                row_skus.append(sku_id)

                curr = node_info.get("e", -1)

            if row_skus:
                rows.append(row_skus)

            # Move south to find the next shelf row
            next_left = graph[current_left].get("s", -1)
            if next_left == -1:
                # Try finding an unvisited left node
                remaining = [ln for ln in left_nodes if ln not in visited_rows and ln not in node_visited]
                next_left = remaining[0] if remaining else -1
            current_left = next_left

        return rows if rows else [[cls.normalize_sku_id(obj["img_path"]) for obj in objects]]


class PlanogramAligner:
    """Performs sequence alignment between observed shelf rows and expected planograms.

    Uses Needleman-Wunsch dynamic programming edit distance to produce global alignment
    between expected and detected SKU sequences, categorizing every slot into:
    - CORRECT: detected == expected
    - MISPLACED: expected item replaced by another known product
    - MISSING: expected item omitted on shelf
    - UNKNOWN: product facing detected but unidentifiable (open-set reject)
    """

    def __init__(
        self,
        match_cost: float = 0.0,
        substitute_cost: float = 1.0,
        gap_cost: float = 1.0
    ) -> None:
        """Initializes alignment penalties.

        Args:
            match_cost: Penalty when observed matches expected (0.0).
            substitute_cost: Penalty for replacing expected with a different SKU (1.0).
            gap_cost: Penalty for insertions (extra/unexpected) or deletions (missing).
        """
        self.match_cost = match_cost
        self.substitute_cost = substitute_cost
        self.gap_cost = gap_cost

    def align_row(
        self,
        expected: List[str],
        detected: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aligns a single shelf row sequence against expected planogram sequence.

        Args:
            expected: Ordered list of expected SKU IDs, e.g. ['Tea/1', 'Tea/2', 'Tea/3'].
            detected: Ordered list of detected item dicts (left-to-right), each having 'sku_id'.

        Returns:
            Dictionary containing:
                - 'alignment': List of aligned pairs:
                    [{
                       'expected_sku': str or None,
                       'detected_sku': str or None,
                       'status': 'CORRECT' | 'MISPLACED' | 'MISSING' | 'UNKNOWN',
                       'detection_info': Dict or None
                    }, ...]
                - 'edit_distance': Total edit distance cost.
                - 'compliance_rate': Percentage of expected SKUs correctly placed [0.0, 100.0].
                - 'summary': Counts for each status category.
        """
        M = len(expected)
        N = len(detected)

        # Dynamic programming matrix
        dp = np.zeros((M + 1, N + 1), dtype=np.float32)

        for i in range(M + 1):
            dp[i, 0] = i * self.gap_cost
        for j in range(N + 1):
            dp[0, j] = j * self.gap_cost

        for i in range(1, M + 1):
            for j in range(1, N + 1):
                exp_sku = expected[i - 1]
                det_sku = detected[j - 1].get("sku_id", "UNKNOWN")

                cost_sub = self.match_cost if exp_sku == det_sku else self.substitute_cost
                dp[i, j] = min(
                    dp[i - 1, j - 1] + cost_sub,  # Match / Substitute
                    dp[i - 1, j] + self.gap_cost,   # Deletion (Missing)
                    dp[i, j - 1] + self.gap_cost    # Insertion (Extra/Misplaced)
                )

        # Backtracking
        i, j = M, N
        alignment: List[Dict[str, Any]] = []

        while i > 0 or j > 0:
            if i > 0 and j > 0:
                exp_sku = expected[i - 1]
                det_item = detected[j - 1]
                det_sku = det_item.get("sku_id", "UNKNOWN")
                cost_sub = self.match_cost if exp_sku == det_sku else self.substitute_cost

                if np.isclose(dp[i, j], dp[i - 1, j - 1] + cost_sub):
                    if det_sku == "UNKNOWN":
                        status = "UNKNOWN"
                    elif exp_sku == det_sku:
                        status = "CORRECT"
                    else:
                        status = "MISPLACED"

                    alignment.append({
                        "expected_sku": exp_sku,
                        "detected_sku": det_sku,
                        "status": status,
                        "detection_info": det_item
                    })
                    i -= 1
                    j -= 1
                    continue

            if i > 0 and np.isclose(dp[i, j], dp[i - 1, j] + self.gap_cost):
                alignment.append({
                    "expected_sku": expected[i - 1],
                    "detected_sku": None,
                    "status": "MISSING",
                    "detection_info": None
                })
                i -= 1
            else:
                det_item = detected[j - 1]
                det_sku = det_item.get("sku_id", "UNKNOWN")
                status = "UNKNOWN" if det_sku == "UNKNOWN" else "MISPLACED"
                alignment.append({
                    "expected_sku": None,
                    "detected_sku": det_sku,
                    "status": status,
                    "detection_info": det_item
                })
                j -= 1

        alignment.reverse()

        # Compute summary metrics
        counts = {"CORRECT": 0, "MISPLACED": 0, "MISSING": 0, "UNKNOWN": 0}
        for item in alignment:
            counts[item["status"]] = counts.get(item["status"], 0) + 1

        compliance_rate = (counts["CORRECT"] / M * 100.0) if M > 0 else 0.0

        return {
            "alignment": alignment,
            "edit_distance": float(dp[M, N]),
            "compliance_rate": round(compliance_rate, 2),
            "summary": counts
        }

    def align_shelf(
        self,
        expected_rows: List[List[str]],
        detected_rows: List[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Aligns multi-row shelf layouts against multi-row planograms.

        Args:
            expected_rows: List of expected rows from top to bottom.
            detected_rows: List of clustered detected rows from top to bottom.

        Returns:
            Dictionary containing:
                - 'row_alignments': Per-row alignment reports.
                - 'overall_compliance_rate': Weighted average compliance score across all shelves.
                - 'total_summary': Aggregated counts of CORRECT, MISPLACED, MISSING, UNKNOWN.
        """
        row_reports = []
        total_expected = sum(len(r) for r in expected_rows)
        total_correct = 0
        total_summary = {"CORRECT": 0, "MISPLACED": 0, "MISSING": 0, "UNKNOWN": 0}

        num_rows = max(len(expected_rows), len(detected_rows))

        for r_idx in range(num_rows):
            exp_row = expected_rows[r_idx] if r_idx < len(expected_rows) else []
            det_row = detected_rows[r_idx] if r_idx < len(detected_rows) else []

            report = self.align_row(exp_row, det_row)
            report["shelf_row_index"] = r_idx + 1
            row_reports.append(report)

            total_correct += report["summary"].get("CORRECT", 0)
            for k in total_summary:
                total_summary[k] += report["summary"].get(k, 0)

        overall_compliance = (total_correct / total_expected * 100.0) if total_expected > 0 else 0.0

        return {
            "row_alignments": row_reports,
            "overall_compliance_rate": round(overall_compliance, 2),
            "total_summary": total_summary
        }
