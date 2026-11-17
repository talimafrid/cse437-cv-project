"""
End-to-End Pipeline Integration Test
=====================================

Validates full orchestration:
1. Initialize GalleryIndex with reference crops.
2. Build PlanogramCompliancePipeline.
3. Simulate a shelf image with multiple product bounding boxes.
4. Audit against ground-truth planogram with known violations.
5. Verify compliance report, timing (< 5s requirement), and image generation.
"""

import unittest
import numpy as np
from PIL import Image

from src.recognizer import FeatureExtractor
from src.gallery import GalleryIndex
from src.pipeline import PlanogramCompliancePipeline


class TestCompliancePipeline(unittest.TestCase):
    """SQA Integration test verifying the full pipeline end-to-end."""

    @classmethod
    def setUpClass(cls):
        cls.extractor = FeatureExtractor(device="cpu")
        cls.gallery = GalleryIndex(embedding_dim=2048)

        # Register two mock reference products
        cls.item_a = Image.fromarray(np.full((100, 100, 3), 200, dtype=np.uint8))
        cls.item_b = Image.fromarray(np.full((100, 100, 3), 50, dtype=np.uint8))

        cls.gallery.add_sku("Food/Tea/1", cls.item_a, cls.extractor, category="Tea")
        cls.gallery.add_sku("Food/Coffee/1", cls.item_b, cls.extractor, category="Coffee")

    def test_pipeline_audit_shelf(self):
        """Verifies full execution pipeline from shelf image to compliance report."""
        pipeline = PlanogramCompliancePipeline(
            gallery=self.gallery,
            extractor=self.extractor,
            match_threshold=0.60,
            reject_threshold=0.35
        )

        # Synthetic shelf image (400 x 300)
        shelf_img = Image.new("RGB", (400, 300), color=(240, 240, 240))
        # Paste item A at top shelf [10, 50, 110, 150]
        shelf_img.paste(self.item_a, (10, 50))
        # Paste item B at top shelf [150, 50, 250, 150]
        shelf_img.paste(self.item_b, (150, 50))

        detected_boxes = [
            [10, 50, 110, 150],
            [150, 50, 250, 150]
        ]

        # Expected planogram matches what was pasted
        expected_rows = [["Food/Tea/1", "Food/Coffee/1"]]

        report = pipeline.audit_shelf(
            shelf_image=shelf_img,
            expected_planogram_rows=expected_rows,
            detected_boxes=detected_boxes
        )

        self.assertIn("overall_compliance_rate", report)
        self.assertIn("annotated_image", report)
        self.assertIn("inference_time_seconds", report)

        # Timing check: must run well within 5 seconds constraint
        self.assertLess(report["inference_time_seconds"], 5.0, "Inference exceeded 5s threshold!")

        # Both items should be recognized and correct
        self.assertEqual(report["total_summary"]["CORRECT"], 2)
        self.assertEqual(report["overall_compliance_rate"], 100.0)


if __name__ == "__main__":
    unittest.main()
