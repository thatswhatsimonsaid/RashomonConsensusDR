"""Intrinsic unfolding: arc length, both manifold axes, and metric direction."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.datasets import make_s_curve, make_swiss_roll

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from redr.intrinsic import (
    balance_axes,
    intrinsic_recovery,
    triplet_order_disagreement,
)
from redr.manifolds import (
    generate_s_curve,
    generate_swiss_roll,
    s_curve_speed,
    swiss_roll_arc_length,
)
from redr.metrics import _sample_triplets


class ManifoldGeneratorTests(unittest.TestCase):
    def test_s_curve_matches_sklearn_and_has_unit_speed(self):
        ours = generate_s_curve(n_samples=400, noise=0.05, random_state=7)
        reference, parameter = make_s_curve(n_samples=400, noise=0.05, random_state=7)
        self.assertTrue(np.allclose(ours["X_raw"], reference))
        self.assertTrue(np.allclose(ours["parameter"], parameter))
        self.assertTrue(np.allclose(s_curve_speed(parameter), 1.0))
        step = 1e-4
        t = np.linspace(-4.0, 4.0, 9)
        for value in t:
            a = np.array([np.sin(value), np.sign(value) * (np.cos(value) - 1.0)])
            b = np.array(
                [
                    np.sin(value + step),
                    np.sign(value + step) * (np.cos(value + step) - 1.0),
                ]
            )
            self.assertAlmostEqual(np.linalg.norm(b - a) / step, 1.0, places=3)

    def test_swiss_roll_matches_sklearn_and_arc_length_speed(self):
        ours = generate_swiss_roll(n_samples=300, noise=0.15, random_state=3)
        reference, parameter = make_swiss_roll(n_samples=300, noise=0.15, random_state=3)
        self.assertTrue(np.allclose(ours["X_raw"], reference))
        self.assertTrue(np.allclose(ours["parameter"], parameter))
        t = np.linspace(5.0, 12.0, 6)
        arc = swiss_roll_arc_length(t)
        step = 1e-4
        for value, length in zip(t, arc):
            numerical = (swiss_roll_arc_length(value + step) - length) / step
            self.assertAlmostEqual(numerical, np.sqrt(1.0 + value**2), places=3)
        gaps = np.diff(np.sort(ours["arc_length"]))
        raw_gaps = np.diff(np.sort(ours["parameter"]))
        self.assertFalse(np.allclose(gaps, raw_gaps))

    def test_row_correspondence(self):
        data = generate_s_curve(80, noise=0.0, random_state=1)
        self.assertEqual(data["intrinsic"].shape, (80, 2))
        self.assertTrue(np.allclose(data["intrinsic"][:, 0], data["arc_length"]))
        self.assertTrue(np.allclose(data["intrinsic"][:, 1], data["height"]))
        self.assertTrue(np.allclose(data["X_clean"][:, 1], data["height"]))


class IntrinsicMetricTests(unittest.TestCase):
    def test_perfect_unfolding_scores_perfectly_and_is_pose_invariant(self):
        data = generate_s_curve(120, noise=0.0, random_state=2)
        triplets = _sample_triplets(data["n"], 1500, np.random.default_rng(0))
        rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
        isometric = 2.5 * data["intrinsic"] @ rotation + np.array([4.0, -3.0])
        balanced = balance_axes(data["intrinsic"])
        posed_balanced = 0.4 * balanced @ rotation + np.array([-2.0, 8.0])
        iso_scores = intrinsic_recovery(data["intrinsic"], isometric, triplets, k=8)
        bal_scores = intrinsic_recovery(data["intrinsic"], posed_balanced, triplets, k=8)
        self.assertAlmostEqual(iso_scores["intrinsic_triplet_isometric"], 1.0, places=6)
        self.assertAlmostEqual(bal_scores["intrinsic_triplet_balanced"], 1.0, places=6)
        self.assertLess(bal_scores["intrinsic_procrustes"], 1e-6)
        reflected = isometric * np.array([1.0, -1.0])
        reflected_scores = intrinsic_recovery(data["intrinsic"], reflected, triplets, k=8)
        self.assertAlmostEqual(
            reflected_scores["intrinsic_triplet_isometric"], 1.0, places=6
        )

    def test_collapsed_height_is_not_full_recovery(self):
        data = generate_s_curve(160, noise=0.0, random_state=4)
        triplets = _sample_triplets(data["n"], 2000, np.random.default_rng(1))
        arc_only = np.column_stack(
            [data["arc_length"], np.zeros(data["n"])]
        )
        scores = intrinsic_recovery(data["intrinsic"], arc_only, triplets, k=10)
        self.assertGreater(scores["arc_triplet_accuracy"], 0.99)
        self.assertLess(scores["height_triplet_accuracy"], 0.6)
        self.assertLess(
            scores["intrinsic_triplet_balanced"],
            scores["arc_triplet_accuracy"] - 0.05,
        )

    def test_order_disagreement_is_between_zero_and_one(self):
        data = generate_s_curve(200, noise=0.05, random_state=1000)
        triplets = _sample_triplets(data["n"], 4000, np.random.default_rng(5000))
        rate = triplet_order_disagreement(data["intrinsic"], triplets)
        self.assertGreater(rate, 0.0)
        self.assertLess(rate, 0.5)


if __name__ == "__main__":
    unittest.main()
