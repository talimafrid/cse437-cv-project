"""
F5 - Synthetic Violation Injection Module
==========================================

Generates controlled non-compliant test scenarios by perturbing ground-truth
planograms. This module enables rigorous SQA evaluation of the planogram alignment
engine under known violation ground truth.

Supported Mutation Operations:
1. SWAP: Exchanges positions of two SKUs within or between shelves (misplaced facing).
2. REMOVE: Deletes a required SKU facing (out-of-stock / missing product).
3. INSERT: Adds an unauthorized or foreign SKU into a shelf sequence (rogue item).
4. REPLACE: Substitutes an expected SKU with another candidate from a SKU pool.

Every perturbation operation is logged with exact ground-truth labels for metric computation.
"""

from typing import List, Dict, Tuple, Any, Optional
import copy
import random


class ViolationInjector:
    """Generates synthetic non-compliant planograms from valid layouts.

    Attributes:
        mutation_rate (float): Target fraction of slots to perturb [0.05, 0.50].
        operations (List[str]): Allowed operations ('swap', 'remove', 'insert', 'replace').
        sku_pool (List[str]): Pool of valid SKU identifiers for replacements/insertions.
        seed (Optional[int]): Random seed for deterministic reproducibility.
    """

    def __init__(
        self,
        mutation_rate: float = 0.15,
        operations: Optional[List[str]] = None,
        sku_pool: Optional[List[str]] = None,
        seed: Optional[int] = 42
    ) -> None:
        """Initializes the violation injector.

        Args:
            mutation_rate: Probability/proportion of mutations applied to planogram (default 15%).
            operations: Subset of ['swap', 'remove', 'insert', 'replace'].
            sku_pool: List of reference SKU IDs to pull from when inserting/replacing.
            seed: Random seed for reproducible experiment generation.

        Raises:
            ValueError: If mutation_rate is out of valid bounds [0.0, 1.0].
        """
        if not (0.0 <= mutation_rate <= 1.0):
            raise ValueError(f"mutation_rate must be in [0.0, 1.0], got {mutation_rate}")

        self.mutation_rate = mutation_rate
        self.operations = operations or ["swap", "remove", "insert", "replace"]
        self.sku_pool = sku_pool or [
            f"Food/Snacks/{i}" for i in range(1, 50)
        ] + [
            f"Food/Tea/{i}" for i in range(1, 50)
        ]
        self.rng = random.Random(seed)

    def inject_violations(
        self,
        planogram_rows: List[List[str]]
    ) -> Tuple[List[List[str]], List[Dict[str, Any]]]:
        """Applies random perturbations across multi-shelf planogram rows.

        Args:
            planogram_rows: Original ground-truth planogram rows (top-to-bottom, left-to-right).

        Returns:
            Tuple of:
                - Perturbed planogram rows (List[List[str]])
                - Mutation log (List[Dict[str, Any]]) recording exact edits for verification:
                  [{
                     'operation': 'swap' | 'remove' | 'insert' | 'replace',
                     'row_idx': int,
                     'col_idx': int,
                     'details': str
                  }, ...]

        Raises:
            ValueError: If input planogram_rows is empty.
        """
        if not planogram_rows:
            raise ValueError("Cannot inject violations into an empty planogram.")

        # Deep copy to ensure original planogram remains pristine
        perturbed: List[List[str]] = [list(r) for r in planogram_rows]
        mutation_log: List[Dict[str, Any]] = []

        total_slots = sum(len(r) for r in perturbed)
        if total_slots == 0:
            return perturbed, []

        num_mutations = max(1, int(round(total_slots * self.mutation_rate)))

        for _ in range(num_mutations):
            op = self.rng.choice(self.operations)
            # Pick a non-empty row if available
            available_rows = [i for i, r in enumerate(perturbed) if len(r) > 0]
            if not available_rows:
                break

            r_idx = self.rng.choice(available_rows)
            row = perturbed[r_idx]

            if op == "swap":
                if len(row) >= 2:
                    idx1, idx2 = self.rng.sample(range(len(row)), 2)
                    item1, item2 = row[idx1], row[idx2]
                    row[idx1], row[idx2] = item2, item1
                    mutation_log.append({
                        "operation": "swap",
                        "row_idx": r_idx,
                        "col_idx": idx1,
                        "details": f"Swapped SKU '{item1}' (col {idx1}) with '{item2}' (col {idx2})"
                    })
                elif len(available_rows) >= 2:
                    # Cross-row swap if single-item rows
                    r2_idx = self.rng.choice([i for i in available_rows if i != r_idx])
                    idx1 = 0
                    idx2 = self.rng.randrange(len(perturbed[r2_idx]))
                    item1, item2 = perturbed[r_idx][idx1], perturbed[r2_idx][idx2]
                    perturbed[r_idx][idx1], perturbed[r2_idx][idx2] = item2, item1
                    mutation_log.append({
                        "operation": "swap",
                        "row_idx": r_idx,
                        "col_idx": idx1,
                        "details": f"Cross-shelf swapped row {r_idx} item '{item1}' with row {r2_idx} item '{item2}'"
                    })

            elif op == "remove":
                if len(row) > 0:
                    del_idx = self.rng.randrange(len(row))
                    removed_item = row.pop(del_idx)
                    mutation_log.append({
                        "operation": "remove",
                        "row_idx": r_idx,
                        "col_idx": del_idx,
                        "details": f"Removed SKU '{removed_item}' at position {del_idx}"
                    })

            elif op == "insert":
                ins_idx = self.rng.randint(0, len(row))
                new_sku = self.rng.choice(self.sku_pool)
                row.insert(ins_idx, new_sku)
                mutation_log.append({
                    "operation": "insert",
                    "row_idx": r_idx,
                    "col_idx": ins_idx,
                    "details": f"Inserted foreign SKU '{new_sku}' at position {ins_idx}"
                })

            elif op == "replace":
                if len(row) > 0:
                    rep_idx = self.rng.randrange(len(row))
                    old_sku = row[rep_idx]
                    new_sku = self.rng.choice(self.sku_pool)
                    row[rep_idx] = new_sku
                    mutation_log.append({
                        "operation": "replace",
                        "row_idx": r_idx,
                        "col_idx": rep_idx,
                        "details": f"Replaced SKU '{old_sku}' with foreign SKU '{new_sku}' at position {rep_idx}"
                    })

        return perturbed, mutation_log
