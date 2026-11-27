"""Rashomon membership stays inside one objective configuration."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from redr.cross_objective import (  # noqa: E402
    apply_membership,
    config_id,
    is_default,
    iter_jobs,
    load_config,
    neg_spec_param,
    pair_relation,
    quality_weights,
)


class CrossObjectiveTests(unittest.TestCase):
    def test_neighbor_counts_are_different_objectives(self):
        self.assertNotEqual(config_id("umap", {"k": 15}), config_id("umap", {"k": 50}))

    def test_spectrum_changes_the_negative_tsne_id(self):
        left = config_id("neg_tsne", {"k": 15, "spectrum": 0.0})
        right = config_id("neg_tsne", {"k": 15, "spectrum": 1.0})
        self.assertNotEqual(left, right)

    def test_vendor_spectrum_map(self):
        self.assertAlmostEqual(neg_spec_param(240, 1.0), 1.0)
        self.assertAlmostEqual(neg_spec_param(240, 0.0), 5 * 24000 / 57600)

    def test_band_uses_the_configuration_best_and_does_not_pool(self):
        frame = pd.DataFrame(
            {
                "config_id": ["umap|k=15", "umap|k=15", "umap|k=50", "umap|k=50"],
                "native_loss": [1.0, 1.04, 10.0, 10.7],
            }
        )
        marked, inactive, span = apply_membership(frame, 0.05)
        kept = marked["in_rashomon"].tolist()
        self.assertEqual(kept, [True, True, True, False])
        self.assertFalse(inactive)
        self.assertAlmostEqual(span, 0.07)

    def test_inactive_when_every_seed_is_inside_its_own_band(self):
        frame = pd.DataFrame(
            {
                "config_id": ["umap|k=15", "umap|k=15", "pca"],
                "native_loss": [1.0, 1.01, 0.0],
            }
        )
        _marked, inactive, span = apply_membership(frame, 0.05)
        self.assertTrue(inactive)
        self.assertLess(span, 0.05)

    def test_relations(self):
        self.assertEqual(pair_relation("umap", "umap|k=15", "umap", "umap|k=15"), "within_config")
        self.assertEqual(pair_relation("umap", "umap|k=15", "umap", "umap|k=50"), "between_config")
        self.assertEqual(pair_relation("umap", "umap|k=15", "pca", "pca"), "between_algorithm")

    def test_defaults_are_in_the_grid(self):
        config = load_config()
        jobs = iter_jobs(config, [0, 1, 2])
        for method in config["methods"]:
            self.assertTrue(is_default(method.get("default"), method.get("default")))
            defaults = [job for job in jobs if job["method"] == method["name"] and job["is_default"]]
            self.assertGreaterEqual(len(defaults), 1)
        umap_k = {job["k"] for job in jobs if job["method"] == "umap"}
        self.assertEqual(umap_k, {5, 15, 30, 50})

    def test_quality_weights_prefer_lower_loss(self):
        weights = quality_weights(np.array([0.0, 1.0, 2.0]), beta=1.0)
        self.assertGreater(weights[0], weights[-1])
        self.assertAlmostEqual(float(weights.sum()), 1.0)


if __name__ == "__main__":
    unittest.main()
