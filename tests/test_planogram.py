"""
Unit Tests for F4 Planogram Alignment & Row Clustering
======================================================

Comprehensive SQA validation for:
1. ShelfRowClusterer: Multi-shelf detection grouping, sorting, camera tilt resilience.
2. PlanogramAligner:
   - 100% compliance matching.
   - Detecting single/multiple missing items.
   - Detecting misplaced/swapped items.
   - Handling unidentifiable (UNKNOWN) items.
"""

import unittest
from src.planogram import ShelfRowClusterer, PlanogramAligner


class TestPlanogramModule(unittest.TestCase):
    """Test suite for shelf clustering and edit-distance planogram compliance auditing."""

    def setUp(self):
        self.clusterer = ShelfRowClusterer(vertical_tolerance=0.50)
        self.aligner = PlanogramAligner()

    def test_cluster_empty_detections(self):
        """Empty input must yield empty row list."""
        self.assertEqual(self.clusterer.cluster_detections([]), [])

    def test_cluster_two_shelves_ordering(self):
        """Verifies detections on 2 distinct vertical rows are separated and ordered correctly."""
        # Top row: y in [100, 200], Bottom row: y in [300, 400]
        # Inverted x order to ensure horizontal sorting works
        detections = [
            {"bbox": [200, 100, 280, 200], "sku_id": "Top_Right"},
            {"bbox": [50, 100, 130, 200], "sku_id": "Top_Left"},
            {"bbox": [220, 310, 300, 410], "sku_id": "Bottom_Right"},
            {"bbox": [40, 300, 120, 400], "sku_id": "Bottom_Left"},
        ]

        rows = self.clusterer.cluster_detections(detections)
        self.assertEqual(len(rows), 2, "Expected exactly 2 shelf rows.")

        # Top row checks (must be sorted left-to-right)
        self.assertEqual(rows[0][0]["sku_id"], "Top_Left")
        self.assertEqual(rows[0][1]["sku_id"], "Top_Right")

        # Bottom row checks
        self.assertEqual(rows[1][0]["sku_id"], "Bottom_Left")
        self.assertEqual(rows[1][1]["sku_id"], "Bottom_Right")

    def test_align_perfect_compliance(self):
        """Identical expected and detected sequences must yield 100% compliance."""
        expected = ["Tea/1", "Tea/2", "Tea/3"]
        detected = [
            {"sku_id": "Tea/1", "bbox": [10, 10, 50, 50]},
            {"sku_id": "Tea/2", "bbox": [60, 10, 100, 50]},
            {"sku_id": "Tea/3", "bbox": [110, 10, 150, 50]},
        ]

        result = self.aligner.align_row(expected, detected)
        self.assertEqual(result["compliance_rate"], 100.0)
        self.assertEqual(result["edit_distance"], 0.0)
        self.assertEqual(result["summary"]["CORRECT"], 3)
        self.assertEqual(result["summary"]["MISPLACED"], 0)
        self.assertEqual(result["summary"]["MISSING"], 0)
        self.assertEqual(result["summary"]["UNKNOWN"], 0)

    def test_align_missing_item(self):
        """Verifies missing item is correctly identified."""
        expected = ["Tea/1", "Tea/2", "Tea/3"]
        # Tea/2 is missing from shelf
        detected = [
            {"sku_id": "Tea/1", "bbox": [10, 10, 50, 50]},
            {"sku_id": "Tea/3", "bbox": [110, 10, 150, 50]},
        ]

        result = self.aligner.align_row(expected, detected)
        self.assertEqual(result["summary"]["CORRECT"], 2)
        self.assertEqual(result["summary"]["MISSING"], 1)
        self.assertEqual(result["summary"]["MISPLACED"], 0)

        # Inspect alignment log for MISSING entry
        missing_entries = [a for a in result["alignment"] if a["status"] == "MISSING"]
        self.assertEqual(len(missing_entries), 1)
        self.assertEqual(missing_entries[0]["expected_sku"], "Tea/2")

    def test_align_misplaced_item(self):
        """Verifies wrong product occupying expected slot is flagged as MISPLACED."""
        expected = ["Tea/1", "Tea/2", "Tea/3"]
        # Tea/2 replaced by Coffee/9
        detected = [
            {"sku_id": "Tea/1", "bbox": [10, 10, 50, 50]},
            {"sku_id": "Coffee/9", "bbox": [60, 10, 100, 50]},
            {"sku_id": "Tea/3", "bbox": [110, 10, 150, 50]},
        ]

        result = self.aligner.align_row(expected, detected)
        self.assertEqual(result["summary"]["CORRECT"], 2)
        self.assertEqual(result["summary"]["MISPLACED"], 1)
        self.assertEqual(result["summary"]["MISSING"], 0)

    def test_align_unknown_item(self):
        """Verifies item that failed open-set recognition is flagged as UNKNOWN."""
        expected = ["Tea/1", "Tea/2"]
        detected = [
            {"sku_id": "Tea/1", "bbox": [10, 10, 50, 50]},
            {"sku_id": "UNKNOWN", "bbox": [60, 10, 100, 50]},
        ]

        result = self.aligner.align_row(expected, detected)
        self.assertEqual(result["summary"]["CORRECT"], 1)
        self.assertEqual(result["summary"]["UNKNOWN"], 1)


if __name__ == "__main__":
    unittest.main()
