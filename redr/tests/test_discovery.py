"""Selection without latent labels, and a classical MDS baseline."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from redr.baselines import classical_mds
from redr.discovery import observed_matrix
from redr.diversity import procrustes_rmse
from redr.product import generate_product
from redr.selection_rules import diversity_first, pareto_front, union_of_objectives


class BaselineTests(unittest.TestCase):
    def test_classical_mds_recovers_a_plane(self):
        rng = np.random.default_rng(0)
        cloud = rng.normal(size=(40, 2))
        distance = np.linalg.norm(cloud[:, None, :] - cloud[None, :, :], axis=2)
        recovered = classical_mds(distance, n_components=2)
        self.assertLess(procrustes_rmse(recovered, cloud), 1e-6)

    def test_rotation_preserves_distances_and_mixes_columns(self):
        data = generate_product(30, n_nuisance=4, noise=0.0, random_state=1)
        rotated = observed_matrix(data, "rotated", mix_seed=4)
        original = np.linalg.norm(data["X"][0] - data["X"][1])
        mixed = np.linalg.norm(rotated[0] - rotated[1])
        self.assertAlmostEqual(original, mixed, places=6)
        self.assertGreater(np.linalg.norm(rotated[:, 0] - data["X"][:, 0]), 0.1)


class SelectionTests(unittest.TestCase):
    def test_union_keeps_a_view_the_global_loss_drops(self):
        frame = pd.DataFrame(
            {
                "embedding_id": ["global", "subset"],
                "multiscale_loss": [0.20, 0.80],
                "subset_loss": [0.70, 0.20],
            }
        )
        chosen = union_of_objectives(frame, ["multiscale_loss", "subset_loss"], epsilon=0.05)
        self.assertEqual(set(chosen["embedding_id"]), {"global", "subset"})

    def test_pareto_drops_a_dominated_embedding(self):
        frame = pd.DataFrame(
            {
                "embedding_id": ["good", "bad", "trade"],
                "multiscale_loss": [0.2, 0.9, 0.5],
                "subset_loss": [0.5, 0.9, 0.2],
            }
        )
        chosen = pareto_front(frame, ["multiscale_loss", "subset_loss"])
        self.assertEqual(set(chosen["embedding_id"]), {"good", "trade"})

    def test_diversity_first_can_prefer_an_unrelated_cloud(self):
        rng = np.random.default_rng(0)
        base = rng.normal(size=(30, 2))
        twin = base + 0.01 * rng.normal(size=base.shape)
        noise = rng.normal(size=(30, 2))
        frame = pd.DataFrame(
            {
                "embedding_id": ["base", "twin", "noise"],
                "multiscale_loss": [0.1, 0.1, 0.9],
            }
        )
        embeddings = {"base": base, "twin": twin, "noise": noise}
        chosen = diversity_first(frame, embeddings, k=5, keep_max=2)
        self.assertIn("noise", set(chosen["embedding_id"]))
        self.assertNotIn("twin", set(chosen["embedding_id"]))


if __name__ == "__main__":
    unittest.main()
