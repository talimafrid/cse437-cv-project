"""
Unit Tests for F2/F3 SKU Recognition & F6 Dynamic Gallery Management
===================================================================

Standard unittest suite testable with both standard Python `unittest` and `pytest`.

Tests:
1. FeatureExtractor embedding shape and unit L2-norm verification.
2. SKUMatcher dual-threshold open-set logic (CONFIRMED, LOW_CONFIDENCE, UNKNOWN).
3. Faiss-backed GalleryIndex dynamic registration (F6 Add-a-SKU).
4. Save and load persistence roundtrip.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
from PIL import Image

from src.recognizer import FeatureExtractor, SKUMatcher
from src.gallery import GalleryIndex


class TestRecognizerAndGallery(unittest.TestCase):
    """SQA Test Suite for visual embedding extraction, matching, and dynamic gallery indexing."""

    @classmethod
    def setUpClass(cls):
        """Initializes shared FeatureExtractor once for all test cases."""
        cls.extractor = FeatureExtractor(backbone="resnet50", device="cpu")

    def setUp(self):
        """Creates synthetic RGB test images."""
        arr = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        self.dummy_rgb_image = Image.fromarray(arr)

    def test_feature_extractor_shape_and_norm(self):
        """Verifies that extracted embeddings have dimension 2048 and L2 norm == 1.0."""
        feat = self.extractor.extract_single(self.dummy_rgb_image)

        self.assertIsInstance(feat, np.ndarray)
        self.assertEqual(feat.shape, (2048,), f"Expected shape (2048,), got {feat.shape}")

        norm = np.linalg.norm(feat)
        self.assertTrue(np.isclose(norm, 1.0, atol=1e-5), f"Embedding not normalized: norm={norm}")

    def test_feature_extractor_batch(self):
        """Verifies batch feature extraction dimensions and normalization."""
        images = [self.dummy_rgb_image for _ in range(5)]
        feats = self.extractor.extract_batch(images, batch_size=2)

        self.assertEqual(feats.shape, (5, 2048))
        norms = np.linalg.norm(feats, axis=1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-5))

    def test_dynamic_add_sku_and_search(self):
        """Verifies F6 Add-a-SKU dynamic indexing without retraining."""
        gallery = GalleryIndex(embedding_dim=2048)
        self.assertEqual(len(gallery), 0)

        # Register new SKU dynamically
        idx = gallery.add_sku(
            sku_id="Food/Tea/1",
            image_path_or_crop=self.dummy_rgb_image,
            extractor=self.extractor,
            category="Food/Tea"
        )
        self.assertEqual(idx, 0)
        self.assertEqual(len(gallery), 1)

        # Query matching itself must yield score ~ 1.0
        query_emb = self.extractor.extract_single(self.dummy_rgb_image).reshape(1, -1)
        results = gallery.search(query_emb, top_k=1)[0]

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["sku_id"], "Food/Tea/1")
        self.assertTrue(np.isclose(results[0]["score"], 1.0, atol=1e-4))

    def test_sku_matcher_open_set_thresholds(self):
        """Verifies open-set rejection boundaries: CONFIRMED, LOW_CONFIDENCE, UNKNOWN."""
        matcher = SKUMatcher(match_threshold=0.65, reject_threshold=0.40)
        gallery = GalleryIndex(embedding_dim=2048)

        # Synthetic normalized reference vector
        ref_vec = np.zeros((1, 2048), dtype=np.float32)
        ref_vec[0, 0] = 1.0
        gallery.index.add(ref_vec)
        gallery.metadata.append({"index_id": 0, "sku_id": "TargetSKU", "category": "Test"})

        # Case 1: Identical vector -> cosine similarity 1.0 -> CONFIRMED
        res1 = matcher.match(ref_vec, gallery)
        self.assertEqual(res1["status"], "CONFIRMED")
        self.assertEqual(res1["sku_id"], "TargetSKU")
        self.assertTrue(np.isclose(res1["confidence"], 1.0, atol=1e-4))

        # Case 2: Intermediate similarity 0.50 -> LOW_CONFIDENCE
        vec_mid = np.zeros((1, 2048), dtype=np.float32)
        vec_mid[0, 0] = 0.50
        vec_mid[0, 1] = np.sqrt(1.0 - 0.50**2)
        res2 = matcher.match(vec_mid, gallery)
        self.assertEqual(res2["status"], "LOW_CONFIDENCE")
        self.assertEqual(res2["sku_id"], "TargetSKU")
        self.assertTrue(np.isclose(res2["confidence"], 0.50, atol=1e-4))

        # Case 3: Orthogonal vector -> similarity 0.0 -> UNKNOWN (Open-Set Reject)
        vec_zero = np.zeros((1, 2048), dtype=np.float32)
        vec_zero[0, 1] = 1.0
        res3 = matcher.match(vec_zero, gallery)
        self.assertEqual(res3["status"], "UNKNOWN")
        self.assertEqual(res3["sku_id"], "UNKNOWN")
        self.assertTrue(np.isclose(res3["confidence"], 0.0, atol=1e-4))

    def test_gallery_persistence_roundtrip(self):
        """Verifies saving and loading gallery index and metadata."""
        temp_dir = tempfile.mkdtemp()
        try:
            gallery = GalleryIndex(embedding_dim=2048)
            gallery.add_sku("Item/A", self.dummy_rgb_image, self.extractor)
            gallery.save(temp_dir)

            loaded_gallery = GalleryIndex(embedding_dim=2048)
            loaded_gallery.load(temp_dir)

            self.assertEqual(len(loaded_gallery), 1)
            self.assertEqual(loaded_gallery.metadata[0]["sku_id"], "Item/A")

            # Search against loaded gallery
            query_emb = self.extractor.extract_single(self.dummy_rgb_image).reshape(1, -1)
            res = loaded_gallery.search(query_emb, top_k=1)[0]
            self.assertEqual(res[0]["sku_id"], "Item/A")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
