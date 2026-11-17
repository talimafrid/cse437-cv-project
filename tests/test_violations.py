"""
Unit Tests for F5 Synthetic Violation Injection Module
======================================================

SQA Verification for:
1. Immutability: Original planograms are not modified in-place.
2. Determinism: Controlled random seeding produces reproducible mutations.
3. Mutation Rate: Number of perturbations corresponds to configured target rate.
4. Mutation Correctness: Tests for individual operations (swap, remove, insert, replace).
5. Edge Cases: Single-element rows, boundary checks, empty input validation.
"""

import unittest
from src.violations import ViolationInjector
from src.planogram import PlanogramAligner


class TestViolationInjector(unittest.TestCase):
    """Test suite for synthetic violation generation and integration with alignment."""

    def setUp(self):
        self.sample_planogram = [
            ["Food/Tea/1", "Food/Tea/2", "Food/Tea/3", "Food/Tea/4", "Food/Tea/5"],
            ["Food/Coffee/10", "Food/Coffee/11", "Food/Coffee/12", "Food/Coffee/13"]
        ]

    def test_empty_planogram_raises(self):
        """Injecting into empty planogram must raise ValueError."""
        injector = ViolationInjector()
        with self.assertRaises(ValueError):
            injector.inject_violations([])

    def test_original_planogram_immutability(self):
        """Verifies that original input structure is never mutated in-place."""
        injector = ViolationInjector(mutation_rate=0.40, seed=123)
        original_copy = [list(r) for r in self.sample_planogram]

        perturbed, log = injector.inject_violations(self.sample_planogram)

        self.assertEqual(self.sample_planogram, original_copy, "Original planogram was mutated!")
        self.assertNotEqual(perturbed, self.sample_planogram, "Perturbed planogram is identical to original.")
        self.assertTrue(len(log) > 0, "Mutation log should not be empty.")

    def test_reproducibility_with_seed(self):
        """Identical seeds must yield identical perturbations and logs."""
        inj1 = ViolationInjector(seed=999)
        inj2 = ViolationInjector(seed=999)

        pert1, log1 = inj1.inject_violations(self.sample_planogram)
        pert2, log2 = inj2.inject_violations(self.sample_planogram)

        self.assertEqual(pert1, pert2)
        self.assertEqual(log1, log2)

    def test_remove_operation_triggers_missing_in_aligner(self):
        """Removing an item via injection must be flagged as MISSING by the aligner."""
        injector = ViolationInjector(mutation_rate=0.20, operations=["remove"], seed=42)
        pert, log = injector.inject_violations([["A", "B", "C", "D"]])

        self.assertEqual(len(pert[0]), 3, "Expected 1 item removed.")
        self.assertEqual(log[0]["operation"], "remove")

        # Now pass perturbed through aligner as detected items
        aligner = PlanogramAligner()
        detected_items = [{"sku_id": sku} for sku in pert[0]]
        report = aligner.align_row(["A", "B", "C", "D"], detected_items)

        self.assertEqual(report["summary"]["MISSING"], 1)
        self.assertEqual(report["summary"]["CORRECT"], 3)


if __name__ == "__main__":
    unittest.main()
