"""Dependence clustering and projection rules do not read latent coordinates."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from redr.dependence import (  # noqa: E402
    cluster_columns,
    mutual_information,
    rashomon_groups,
    stack_coordinates,
)
from redr.factors import generate_curves  # noqa: E402
from redr.projections import projection_bank, stress_rashomon_pair  # noqa: E402


class DependenceTests(unittest.TestCase):
    def test_mutual_information_is_symmetric(self):
        rng = np.random.default_rng(0)
        matrix = rng.normal(size=(40, 5))
        scores = mutual_information(matrix, random_state=0)
        self.assertEqual(scores.shape, (5, 5))
        np.testing.assert_allclose(scores, scores.T, atol=1e-8)
        np.testing.assert_allclose(np.diag(scores), 0.0)

    def test_mutual_information_recovers_two_independent_curves(self):
        data = generate_curves(
            n_samples=80,
            n_curves=2,
            noise=0.0,
            n_nuisance=0,
            mixing="aligned",
            random_state=2,
        )
        groups = cluster_columns(data["X_visible"], n_clusters=2, random_state=2)
        self.assertEqual(len(groups), 2)
        grouped = []
        for group in groups:
            self.assertEqual(len(group), 3)
            grouped.append(tuple(sorted(int(index) for index in group)))
        self.assertCountEqual(grouped, [(0, 1, 2), (3, 4, 5)])

    def test_rashomon_band_drops_a_weak_view(self):
        groups = [np.array([0, 1, 2]), np.array([3, 4, 5]), np.array([6, 7, 8])]
        kept = rashomon_groups(groups, [0.20, 0.205, 0.80], epsilon=0.05)
        self.assertEqual(len(kept), 2)

    def test_stack_reduces_three_coordinates_to_two(self):
        rng = np.random.default_rng(1)
        columns = [rng.normal(size=25) for _ in range(3)]
        stacked = stack_coordinates(columns)
        self.assertEqual(stacked.shape, (25, 2))


class ProjectionTests(unittest.TestCase):
    def test_stress_pair_has_two_columns(self):
        rng = np.random.default_rng(3)
        matrix = rng.normal(size=(30, 8))
        coordinates, stresses = projection_bank(matrix, n_projections=4, random_state=3, pca_dim=4)
        self.assertEqual(coordinates.shape, (30, 4))
        paired = stress_rashomon_pair(coordinates, stresses, epsilon=0.05)
        self.assertEqual(paired.shape, (30, 2))
        self.assertTrue(np.isfinite(paired).all())


if __name__ == "__main__":
    unittest.main()
