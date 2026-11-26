"""Regime labels follow the frozen thresholds, not a fitted cutoff."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from redr.natural_diversity import classify_regime, load_config, primary_pool, relative_span  # noqa: E402


def _methods(rows: list[dict], between: float) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["between_disagreement"] = between
    return frame


class NaturalDiversityTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()

    def test_primary_pool_keeps_one_objective(self):
        frame = pd.DataFrame(
            {
                "method": ["umap", "umap", "umap"],
                "k": [15, 5, 15],
                "lambda": [0.0, 0.0, 5.0],
            }
        )
        kept = primary_pool(frame, self.config)
        self.assertEqual(len(kept), 1)
        self.assertEqual(int(kept.iloc[0]["k"]), 15)

    def test_relative_span_matches_the_membership_scale(self):
        self.assertAlmostEqual(relative_span([1.0, 1.04, 1.2]), 0.2)

    def test_exchangeable_copies(self):
        frame = _methods(
            [
                {"method": "umap", "span": 0.01, "mean_trust": 0.95, "within_disagreement": 0.04, "trust_tracks_loss": False},
                {"method": "infonce", "span": 0.02, "mean_trust": 0.94, "within_disagreement": 0.05, "trust_tracks_loss": False},
            ],
            between=0.05,
        )
        label = classify_regime(frame, self.config)
        self.assertEqual(label["primary"], "exchangeable")
        self.assertFalse(label["systematically_bad"])

    def test_contaminated_draws(self):
        frame = _methods(
            [
                {"method": "umap", "span": 0.20, "mean_trust": 0.90, "within_disagreement": 0.10, "trust_tracks_loss": True},
                {"method": "infonce", "span": 0.18, "mean_trust": 0.88, "within_disagreement": 0.10, "trust_tracks_loss": True},
            ],
            between=0.12,
        )
        label = classify_regime(frame, self.config)
        self.assertEqual(label["primary"], "contaminated_within_method")

    def test_systematically_bad_method(self):
        frame = _methods(
            [
                {"method": "umap", "span": 0.01, "mean_trust": 0.94, "within_disagreement": 0.04, "trust_tracks_loss": False},
                {"method": "nce", "span": 0.02, "mean_trust": 0.70, "within_disagreement": 0.04, "trust_tracks_loss": False},
            ],
            between=0.30,
        )
        label = classify_regime(frame, self.config)
        self.assertEqual(label["primary"], "systematically_bad")
        self.assertEqual(label["bad_methods"], "nce")

    def test_span_without_trust_is_not_contamination(self):
        frame = _methods(
            [
                {"method": "umap", "span": 0.20, "mean_trust": 0.90, "within_disagreement": 0.08, "trust_tracks_loss": False},
                {"method": "infonce", "span": 0.01, "mean_trust": 0.91, "within_disagreement": 0.04, "trust_tracks_loss": False},
            ],
            between=0.08,
        )
        label = classify_regime(frame, self.config)
        self.assertEqual(label["primary"], "loss_varies_unstructured")
        self.assertFalse(label["contaminated_within_method"])

    def test_complementary_candidate(self):
        frame = _methods(
            [
                {"method": "umap", "span": 0.01, "mean_trust": 0.90, "within_disagreement": 0.05, "trust_tracks_loss": False},
                {"method": "infonce", "span": 0.02, "mean_trust": 0.89, "within_disagreement": 0.04, "trust_tracks_loss": False},
            ],
            between=0.40,
        )
        label = classify_regime(frame, self.config)
        self.assertEqual(label["primary"], "complementary_candidate")


if __name__ == "__main__":
    unittest.main()
