"""The An and Tang generators and Meta-Spec distance stay well-defined."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from redr.paper_views import (  # noqa: E402
    induced_graph_weights,
    meta_spec_distance,
    mixture_of_gaussians,
    soft_jaccard_distance,
    swiss_roll,
)


class PaperViewTests(unittest.TestCase):
    def test_gaussian_mixture_has_the_paper_sizes(self):
        data = mixture_of_gaussians(random_state=1)
        self.assertEqual(data["X"].shape, (600, 3))
        self.assertEqual([int(np.sum(data["labels"] == label)) for label in (1, 2, 3)], [150, 200, 250])
        self.assertTrue(np.isfinite(data["X"]).all())

    def test_swiss_roll_keeps_the_parameter(self):
        data = swiss_roll(n_samples=40, random_state=2)
        self.assertEqual(data["X"].shape, (40, 3))
        self.assertEqual(data["intrinsic"].shape, (40, 2))
        self.assertGreater(data["t"].max(), data["t"].min())

    def test_meta_spec_distance_is_symmetric(self):
        rng = np.random.default_rng(0)
        cloud = rng.normal(size=(25, 2))
        distance = meta_spec_distance([cloud, cloud[:, ::-1]])
        self.assertEqual(distance.shape, (25, 25))
        np.testing.assert_allclose(distance, distance.T, atol=1e-8)
        self.assertTrue(np.isfinite(distance).all())

    def test_identical_embeddings_have_soft_jaccard_zero(self):
        rng = np.random.default_rng(3)
        cloud = rng.normal(size=(12, 2))
        neighbors = np.argsort(np.linalg.norm(cloud[:, None, :] - cloud[None, :, :], axis=2), axis=1)[:, 1:4]
        weights = induced_graph_weights(cloud, neighbors)
        self.assertAlmostEqual(soft_jaccard_distance(weights, weights), 0.0, places=8)


if __name__ == "__main__":
    unittest.main()
