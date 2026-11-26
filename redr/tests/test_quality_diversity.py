"""The quality and diversity factors move the pools in the intended directions."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from redr.quality_diversity import (  # noqa: E402
    build_pool,
    cells,
    load_config,
    retained,
    selection_scores,
    trimmed,
)


class QualityDiversityTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()

    def _pool(self, quality: str, diversity: str, score_kind: str = "informative"):
        return build_pool(self.config, {"quality": quality, "diversity": diversity, "score_kind": score_kind}, 0)

    def test_grid_covers_the_factorial_and_the_score_slice(self):
        found = {(item["quality"], item["diversity"], item["score_kind"]) for item in cells(self.config)}
        self.assertEqual(len(found), 5 * 4 + 2 * 3)
        self.assertIn(("shared", "complementary", "informative"), found)
        self.assertIn(("mixed", "harmful", "misleading"), found)
        locomds = {tuple(item) for item in self.config["locomds_cells"]}
        self.assertIn(("mixed", "complementary", "informative"), locomds)
        self.assertIn(("high", "identical", "informative"), locomds)

    def test_severe_draws_are_worse_than_high_quality_copies(self):
        high = self._pool("high", "identical")["frame"]["true_global_error"].mean()
        severe = self._pool("severe", "identical")["frame"]["true_global_error"].mean()
        self.assertGreater(severe, high + 0.05)

    def test_complementary_specialists_have_higher_global_error_than_the_intact_method(self):
        frame = self._pool("high", "complementary")["frame"]
        specialist = frame.loc[frame["method"].isin(["m0", "m1"]), "true_global_error"].mean()
        intact = frame.loc[frame["method"].eq("m2"), "true_global_error"].mean()
        self.assertGreater(specialist, intact + 0.05)

    def test_harmful_method_is_worse_than_the_faithful_methods(self):
        frame = self._pool("high", "harmful")["frame"]
        junk = frame.loc[frame["method"].eq("m2"), "true_global_error"].mean()
        faithful = frame.loc[frame["method"].eq("m0"), "true_global_error"].mean()
        self.assertGreater(junk, faithful + 0.1)

    def test_scores_have_the_intended_relationship_to_latent_error(self):
        error = np.linspace(0.05, 0.80, 12)
        rng = np.random.default_rng(4)
        informative = selection_scores(error, "informative", rng)
        weak = selection_scores(error, "weak", np.random.default_rng(4))
        misleading = selection_scores(error, "misleading", rng)
        self.assertGreater(pd.Series(informative).corr(pd.Series(error), method="spearman"), 0.8)
        self.assertGreater(pd.Series(weak).corr(pd.Series(error), method="spearman"), 0.3)
        self.assertLess(pd.Series(misleading).corr(pd.Series(error), method="spearman"), -0.99)

    def test_zero_epsilon_keeps_one_draw_per_method_and_infinity_keeps_all(self):
        frame = self._pool("mixed", "identical")["frame"]
        tight = retained(frame, 0.0)
        wide = retained(frame, np.inf)
        self.assertEqual(len(wide), 12)
        self.assertEqual(set(tight["method"]), {"m0", "m1", "m2"})
        self.assertLessEqual(len(tight), 6)

    def test_trim_drops_the_worst_quarter(self):
        frame = self._pool("mixed", "identical")["frame"]
        kept = trimmed(frame, 0.25)
        self.assertEqual(len(kept), 9)
        self.assertLessEqual(kept["score"].max(), frame["score"].nlargest(3).min())


if __name__ == "__main__":
    unittest.main()
