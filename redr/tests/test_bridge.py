"""Python-to-R weight checks. These run CoMDS on tiny synthetic views."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from redr.comds_bridge import fit_consensus
from redr.diversity import procrustes_rmse


def _views(seed: int = 0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(20, 2))
    return [base, base + rng.normal(scale=0.05, size=base.shape), rng.normal(size=(20, 2))]


class BridgeTests(unittest.TestCase):
    def test_uniform_weights_match_omitted_weights(self):
        views = _views()
        out = Path("/tmp/redr-c2-uniform")
        plain = fit_consensus(views, out / "plain", kind="comds")
        weighted = fit_consensus(
            views, out / "ones", kind="comds", weights=np.ones(len(views))
        )
        self.assertAlmostEqual(plain["stress"], weighted["stress"], places=4)
        self.assertLess(
            procrustes_rmse(plain["embedding"], weighted["embedding"]), 1e-3
        )

    def test_permutation_keeps_weighted_objective(self):
        views = _views(1)
        weights = np.array([1.0, 0.4, 1.6])
        order = np.array([2, 0, 1])
        out = Path("/tmp/redr-c2-permute")
        ids = np.arange(20)
        original = fit_consensus(
            views, out / "original", kind="comds", weights=weights, observation_ids=ids
        )
        permuted = fit_consensus(
            [views[i] for i in order],
            out / "permuted",
            kind="comds",
            weights=weights[order],
            observation_ids=ids,
        )
        self.assertAlmostEqual(original["stress"], permuted["stress"], places=3)
        self.assertLess(
            procrustes_rmse(original["embedding"], permuted["embedding"]), 5e-3
        )

    def test_duplicated_view_with_split_weight(self):
        views = _views(2)
        weights = np.array([1.0, 1.0, 1.0])
        out = Path("/tmp/redr-c2-duplicate")
        original = fit_consensus(views, out / "original", kind="comds", weights=weights)
        duplicated_views = [views[0], views[0], views[1], views[2]]
        duplicated_weights = np.array([0.5, 0.5, 1.0, 1.0])
        duplicated = fit_consensus(
            duplicated_views, out / "duplicated", kind="comds", weights=duplicated_weights
        )
        self.assertLess(
            abs(original["stress"] - duplicated["stress"]), 0.05
        )
        self.assertLess(
            procrustes_rmse(original["embedding"], duplicated["embedding"]), 0.05
        )


if __name__ == "__main__":
    unittest.main()
