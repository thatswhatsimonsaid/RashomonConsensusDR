"""Run the controlled-embedding mechanism pilot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from redr.controlled import plot_controlled, run_scenario, scenarios, summarize_controlled  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replications", type=int, default=25)
    parser.add_argument("--seed0", type=int, default=7000)
    args = parser.parse_args()
    out = REPO / "results" / "checkpoint3" / "controlled_ensembles"
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for scenario in scenarios():
        print(f"scenario {scenario['name']}", flush=True)
        frames.append(run_scenario(scenario, args.replications, args.seed0))
    detail = pd.concat(frames, ignore_index=True)
    detail.to_csv(out / "detail.csv", index=False)
    summary = summarize_controlled(detail)
    summary.to_csv(out / "summary.csv", index=False)
    plot_controlled(summary, out / "figures" / "filter_vs_matched.png")
    view = summary.loc[
        summary["scheme"].eq("uniform")
        & summary["rule"].isin(["rashomon", "matched"])
        & np.isclose(summary["epsilon"].astype(float), 0.05),
        ["scenario", "alignment", "oracle_score", "rule", "mean", "std", "mean_spearman", "mean_true_error"],
    ]
    print(view.to_string(index=False))


if __name__ == "__main__":
    main()
