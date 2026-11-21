"""Small checks for the synthetic-embedding mechanism study."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from redr.controlled import run_scenario


class ControlledTests(unittest.TestCase):
    def test_oracle_filter_keeps_lower_error_than_misleading(self):
        base = {
            "n": 60,
            "n_triplets": 300,
            "n_methods": 3,
            "per_method": 6,
            "scale": 0.2,
            "matched_seeds": 2,
            "geometry": "long",
            "structure": "shared_noise",
            "contamination": 0.0,
        }
        oracle = run_scenario({**base, "name": "oracle", "alignment": "oracle"}, 1, 11)
        misleading = run_scenario(
            {**base, "name": "misleading", "alignment": "misleading"}, 1, 11
        )
        oracle_row = oracle.loc[
            (oracle["rule"] == "rashomon")
            & (oracle["scheme"] == "uniform")
            & np.isclose(oracle["epsilon"], 0.05)
        ].iloc[0]
        misleading_row = misleading.loc[
            (misleading["rule"] == "rashomon")
            & (misleading["scheme"] == "uniform")
            & np.isclose(misleading["epsilon"], 0.05)
        ].iloc[0]
        self.assertGreater(oracle_row["score_error_spearman"], 0.99)
        self.assertLess(misleading_row["score_error_spearman"], -0.99)
        self.assertLess(oracle_row["mean_true_error"], misleading_row["mean_true_error"])
        self.assertLess(oracle_row["n_views"], base["n_methods"] * base["per_method"])


if __name__ == "__main__":
    unittest.main()
