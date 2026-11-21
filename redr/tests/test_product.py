"""Product manifold and selection rules. Latent coordinates are evaluation-only."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from redr.intrinsic import intrinsic_recovery, triplet_order_disagreement
from redr.metrics import _sample_triplets
from redr.product import generate_product
from redr.complementary import select_rule
from redr.selection_rules import diversity_aware, hybrid, method_specific, pooled


class ProductTests(unittest.TestCase):
    def test_square_uses_both_axes_and_rows_match(self):
        data = generate_product(180, aspect=(1.0, 1.0), noise=0.0, n_nuisance=0, random_state=0)
        self.assertEqual(data["intrinsic"].shape, (180, 2))
        self.assertTrue(np.allclose(data["intrinsic"][:, 0], data["u"]))
        triplets = _sample_triplets(180, 2000, np.random.default_rng(0))
        rate = triplet_order_disagreement(data["intrinsic"], triplets)
        self.assertGreater(rate, 0.15)
        perfect = intrinsic_recovery(data["intrinsic"], data["intrinsic"], triplets, k=8)
        self.assertAlmostEqual(perfect["intrinsic_triplet_isometric"], 1.0, places=6)
        self.assertGreater(perfect["intrinsic_triplet_balanced"], 0.99)
        # A perfect two-axis embedding is not a perfect ordering of either axis.
        self.assertLess(perfect["arc_triplet_accuracy"], 0.95)
        self.assertLess(perfect["height_triplet_accuracy"], 0.95)

    def test_one_axis_embedding_is_not_full_recovery(self):
        data = generate_product(160, aspect=(1.0, 1.0), noise=0.0, n_nuisance=0, random_state=1)
        triplets = _sample_triplets(160, 1500, np.random.default_rng(1))
        only_u = np.column_stack([data["u"], np.zeros(data["n"])])
        scores = intrinsic_recovery(data["intrinsic"], only_u, triplets, k=8)
        self.assertGreater(scores["arc_triplet_accuracy"], 0.99)
        self.assertLess(scores["height_triplet_accuracy"], 0.6)
        self.assertLess(scores["intrinsic_triplet_balanced"], 0.9)

    def test_preserve_scale_keeps_a_longer_first_block(self):
        square = generate_product(80, aspect=(1.0, 1.0), noise=0.0, n_nuisance=0, preserve_scale=True, random_state=3)
        long = generate_product(80, aspect=(5.0, 1.0), noise=0.0, n_nuisance=0, preserve_scale=True, random_state=3)
        square_energy = np.sum(square["X"][:, :3] ** 2)
        long_energy = np.sum(long["X"][:, :3] ** 2)
        self.assertGreater(long_energy, 10.0 * square_energy)

    def test_hidden_second_axis_is_absent_from_the_second_block(self):
        shown = generate_product(40, noise=0.0, hide_second=False, random_state=2, n_nuisance=0)
        hidden = generate_product(40, noise=0.0, hide_second=True, random_state=2, n_nuisance=0)
        self.assertFalse(np.allclose(shown["block_v"], hidden["block_v"]))
        self.assertTrue(np.allclose(shown["u"], hidden["u"]))


class SelectionTests(unittest.TestCase):
    def _frame(self):
        return pd.DataFrame(
            {
                "embedding_id": ["a1", "a2", "b1", "c1"],
                "objective_group": ["umap|joint|k15", "umap|joint|k15", "isomap|u|k10", "isomap|v|k10"],
                "reference_loss": [1.0, 1.2, 0.4, 0.5],
                "structural_loss": [0.20, 0.22, 0.80, 0.78],
                "method": ["umap_joint", "umap_joint", "isomap_u", "isomap_v"],
            }
        )

    def test_method_specific_keeps_each_group_and_pooled_uses_one_scale(self):
        frame = self._frame()
        specific = method_specific(frame, 0.05)
        self.assertEqual(set(specific["embedding_id"]), {"a1", "b1", "c1"})
        structural = pooled(frame, 0.05)
        self.assertEqual(set(structural["embedding_id"]), {"a1"})
        both = hybrid(frame, 0.5)
        self.assertIn("a1", set(both["embedding_id"]))
        self.assertNotIn("b1", set(both["embedding_id"]))

    def test_diversity_prefers_a_different_neighborhood(self):
        rng = np.random.default_rng(0)
        base = rng.normal(size=(40, 2))
        twin = base + rng.normal(scale=0.01, size=base.shape)
        other = rng.normal(size=(40, 2))
        frame = pd.DataFrame(
            {
                "embedding_id": ["best", "twin", "other"],
                "objective_group": ["g", "g", "h"],
                "reference_loss": [0.1, 0.1, 0.1],
                "structural_loss": [0.10, 0.11, 0.12],
                "method": ["a", "a", "b"],
            }
        )
        embeddings = {"best": base, "twin": twin, "other": other}
        chosen = diversity_aware(frame, embeddings, epsilon=0.5, k=5, keep_max=2)
        self.assertEqual(set(chosen["embedding_id"]), {"best", "other"})

    def test_best_structural_is_one_row_and_ignores_axis_scores(self):
        frame = self._frame()
        frame["source"] = ["joint", "joint", "u", "v"]
        frame["u_triplet"] = [0.1, 0.99, 0.99, 0.2]
        best = select_rule("best", frame, {})
        self.assertEqual(list(best["embedding_id"]), ["a1"])
        specialist = select_rule("specialist", frame, {})
        self.assertEqual(set(specialist["source"]), {"u", "v"})


if __name__ == "__main__":
    unittest.main()
