"""Reference-loss tests. These import the vendored trainers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from redr.loss_validation import contrastive_reference_loss, pacmap_component_losses
from redr.probe import contrastive_probe
from redr.train import high_d_neighbors, pacmap_pairs


class ReferenceLossTests(unittest.TestCase):
    def test_contrastive_reference_is_repeatable(self):
        rng = np.random.default_rng(0)
        points = rng.normal(size=(30, 4))
        embedding = rng.normal(size=(30, 2))
        neighbors = high_d_neighbors(points, k=5)
        first = contrastive_reference_loss(embedding, neighbors, "umap", n_draws=2)
        second = contrastive_reference_loss(embedding, neighbors, "umap", n_draws=2)
        self.assertEqual(first, second)
        self.assertTrue(np.isfinite(first[0]))
        self.assertGreaterEqual(first[1], 0.0)

    def test_checkpoint1_probe_is_one_reseeded_draw(self):
        rng = np.random.default_rng(1)
        points = rng.normal(size=(30, 4))
        embedding = rng.normal(size=(30, 2))
        neighbors = high_d_neighbors(points, k=5)
        draws = [contrastive_probe(embedding, neighbors, "umap") for _ in range(4)]
        self.assertTrue(np.allclose(draws, draws[0]))
        reference, _std = contrastive_reference_loss(
            embedding, neighbors, "umap", n_draws=2
        )
        self.assertTrue(np.isfinite(reference))

    def test_pacmap_terminal_drops_midnear(self):
        rng = np.random.default_rng(2)
        points = rng.normal(size=(40, 3)).astype(np.float32)
        embedding = rng.normal(size=(40, 2))
        pairs = pacmap_pairs(points, seed=0)
        parts = pacmap_component_losses(embedding, pairs[0], pairs[1], pairs[2], 40)
        self.assertAlmostEqual(
            parts["native_loss"], parts["pacmap_nn"] + parts["pacmap_fp"]
        )
        self.assertAlmostEqual(
            parts["probe_loss"],
            parts["native_loss"] + parts["pacmap_mn"],
        )
        self.assertGreater(parts["pacmap_mn"], 0)


if __name__ == "__main__":
    unittest.main()
