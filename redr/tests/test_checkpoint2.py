"""Checkpoint 2 unit tests that do not call R."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from redr.diversity import pairwise_neighborhood_overlap, procrustes_rmse
from redr.entropy import (
    binary_entropy,
    confidence_matrices,
    entropy_decomposition,
    knn_adjacency,
    vote_probability,
)
from redr.metrics import (
    distance_rank_correlation,
    evaluate_embedding,
    knn_classification_accuracy,
    lcmc,
    neighborhood_preservation,
    scale_adjusted_distance_error,
)
from redr.rashomon import gibbs_weights, mark_by_loss, method_balanced_weights, uniform_weights
from redr.validation import validate_embeddings


class MembershipTests(unittest.TestCase):
    def test_relative_band_and_nested_epsilons(self):
        frame = pd.DataFrame(
            {
                "method": ["umap", "umap", "umap", "pacmap"],
                "native_loss": [1.0, 1.02, 2.0, 5.0],
            }
        )
        small = mark_by_loss(frame, "native_loss", 0.05)
        large = mark_by_loss(frame, "native_loss", 0.5)
        self.assertEqual(small["in_rashomon"].tolist(), [True, True, False, True])
        self.assertTrue(np.all(large.loc[small["in_rashomon"], "in_rashomon"]))
        self.assertLess(
            small["in_rashomon"].sum(),
            mark_by_loss(frame, "native_loss", np.inf)["in_rashomon"].sum(),
        )

    def test_zero_keeps_ties_and_infinity_keeps_finite(self):
        frame = pd.DataFrame(
            {"method": ["umap"] * 4, "native_loss": [1.0, 1.0 + 1e-12, 1.1, np.nan]}
        )
        zero = mark_by_loss(frame, "native_loss", 0.0)
        self.assertEqual(zero["in_rashomon"].tolist(), [True, True, False, False])
        infinite = mark_by_loss(frame, "native_loss", np.inf)
        self.assertEqual(infinite["in_rashomon"].tolist(), [True, True, True, False])

    def test_negative_and_near_zero_reference(self):
        negative = mark_by_loss(
            pd.DataFrame({"method": ["umap"] * 3, "native_loss": [-1.0, -0.99, -0.5]}),
            "native_loss",
            0.05,
        )
        self.assertEqual(negative["in_rashomon"].tolist(), [True, True, False])
        near_zero = mark_by_loss(
            pd.DataFrame({"method": ["umap"] * 2, "native_loss": [0.0, 1.0]}),
            "native_loss",
            0.05,
        )
        self.assertEqual(near_zero["in_rashomon"].tolist(), [True, False])
        self.assertGreater(near_zero["loss_tolerance"].iloc[0], 0)


class WeightTests(unittest.TestCase):
    def test_method_balance_and_gibbs_beta_zero(self):
        frame = pd.DataFrame(
            {
                "method": ["umap", "umap", "pacmap"],
                "probe_loss": [1.0, 2.0, 0.5],
            }
        )
        balanced = method_balanced_weights(frame)
        self.assertAlmostEqual(balanced[frame["method"].eq("umap")].sum(), 0.5)
        self.assertAlmostEqual(balanced[frame["method"].eq("pacmap")].sum(), 0.5)
        gibbs = gibbs_weights(frame, beta=0.0)
        self.assertTrue(np.allclose(gibbs, balanced))
        tilted = gibbs_weights(frame, beta=1.0)
        self.assertGreater(tilted[0], tilted[1])
        self.assertAlmostEqual(tilted.sum(), 1.0, places=5)

    def test_duplicate_preserves_method_mass(self):
        frame = pd.DataFrame(
            {"method": ["umap", "pacmap"], "probe_loss": [1.0, 1.0]}
        )
        original = method_balanced_weights(frame)
        duplicated = pd.DataFrame(
            {"method": ["umap", "umap", "pacmap"], "probe_loss": [1.0, 1.0, 1.0]}
        )
        weights = method_balanced_weights(duplicated)
        self.assertAlmostEqual(weights[duplicated["method"].eq("umap")].sum(), original[0])
        uniform = uniform_weights(duplicated)
        self.assertEqual(uniform.tolist(), [1.0, 1.0, 1.0])


class EntropyTests(unittest.TestCase):
    def test_identical_embeddings_have_zero_entropy(self):
        rng = np.random.default_rng(0)
        cloud = rng.normal(size=(25, 2))
        probability = vote_probability([cloud, cloud.copy()], k=5)
        entropy = binary_entropy(probability)
        self.assertTrue(np.allclose(entropy, 0))
        adjacency = knn_adjacency(cloud, 5)
        self.assertEqual(adjacency.sum(), 25 * 5)
        self.assertTrue(np.all(np.diag(adjacency) == 0))

    def test_decomposition_identity(self):
        rng = np.random.default_rng(1)
        embeddings = [rng.normal(size=(20, 2)) + shift for shift in (0, 0.2, 3, 3.4)]
        methods = ["umap", "umap", "pacmap", "pacmap"]
        parts = entropy_decomposition(embeddings, methods, k=4)
        total = parts["total_entropy"]
        within = parts["within_entropy"]
        between = parts["between_information"]
        self.assertTrue(np.allclose(total, within + between, atol=1e-8))
        self.assertGreater(parts["mean_between_information"], 0)

    def test_confidence_matrices_are_symmetric_and_distinct(self):
        rng = np.random.default_rng(2)
        probability = vote_probability(
            [rng.normal(size=(18, 2)), rng.normal(size=(18, 2)) + 1], k=4
        )
        matrices = confidence_matrices(probability, eta=0.05, floor=1e-3)
        for name, matrix in matrices.items():
            self.assertTrue(np.allclose(matrix, matrix.T), name)
            self.assertTrue(np.all(np.diag(matrix) == 0), name)
            self.assertTrue(np.all(matrix >= 0), name)
        self.assertFalse(
            np.allclose(
                matrices["entropy_confidence"], matrices["positive_consensus"]
            )
        )


class MetricTests(unittest.TestCase):
    def test_identity_and_similarity_invariance(self):
        rng = np.random.default_rng(3)
        cloud = rng.normal(size=(40, 4))
        labels = np.array([0] * 20 + [1] * 20)
        embedding = cloud[:, :2] + np.array([0.0, 2.0])
        base = evaluate_embedding(cloud, embedding, labels, k=5)
        rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
        transformed = [
            embedding + np.array([5.0, -2.0]),
            embedding @ rotation,
            embedding @ np.array([[1.0, 0.0], [0.0, -1.0]]),
            embedding * 3.5,
        ]
        for other in transformed:
            scores = evaluate_embedding(cloud, other, labels, k=5)
            for key in (
                "trustworthiness",
                "continuity",
                "neighborhood_preservation",
                "lcmc",
                "triplet_accuracy",
                "silhouette",
            ):
                self.assertAlmostEqual(scores[key], base[key], places=6, msg=key)
        self.assertAlmostEqual(neighborhood_preservation(embedding, embedding.copy(), 5), 1.0)
        self.assertAlmostEqual(lcmc(cloud, cloud, 5), 1.0 - 5 / 39)
        self.assertAlmostEqual(distance_rank_correlation(embedding, embedding * 4), 1.0)
        self.assertAlmostEqual(scale_adjusted_distance_error(cloud[:, :2], embedding), 0.0, places=6)
        separated = np.vstack(
            [np.random.default_rng(5).normal(loc=-3, size=(20, 2)),
             np.random.default_rng(6).normal(loc=3, size=(20, 2))]
        )
        self.assertGreater(knn_classification_accuracy(separated, labels, k=5), 0.9)
        self.assertLess(procrustes_rmse(embedding, embedding * -2 + 4), 1e-6)

    def test_validation_rejects_bad_stacks(self):
        good = np.random.default_rng(4).normal(size=(10, 2))
        validate_embeddings([good, good + 1], observation_ids=np.arange(10))
        with self.assertRaises(ValueError):
            validate_embeddings([good, good[:8]])
        with self.assertRaises(ValueError):
            validate_embeddings([np.full((10, 2), np.nan)])
        with self.assertRaises(ValueError):
            validate_embeddings([np.zeros((10, 2))])
        overlap = pairwise_neighborhood_overlap([good, good.copy()], k=3)
        self.assertAlmostEqual(overlap[0, 1], 1.0)


if __name__ == "__main__":
    unittest.main()
