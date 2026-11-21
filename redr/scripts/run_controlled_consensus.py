"""Run the small CoMDS mechanism grid.

    PYTHONPATH=redr python redr/scripts/run_controlled_consensus.py --scenarios comp_square_noisy --reps 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "redr"))

from redr.controlled_consensus import run_scenario, scenarios  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", default="")
    parser.add_argument("--reps", type=int, default=0)
    parser.add_argument("--kinds", default="procrustes,comds,locomds")
    parser.add_argument("--seed0", type=int, default=9000)
    args = parser.parse_args()
    chosen = scenarios()
    if args.scenarios:
        names = set(args.scenarios.split(","))
        chosen = [item for item in chosen if item["name"] in names]
    if args.reps:
        for item in chosen:
            item["n_replications"] = args.reps
    kinds = [part for part in args.kinds.split(",") if part]
    out = REPO / "results" / "checkpoint4" / "controlled_consensus"
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    dest = out / "detail.csv"
    if dest.exists():
        frames.append(pd.read_csv(dest))
    done = set()
    if frames:
        done = set(zip(frames[0]["scenario"], frames[0]["replication"], frames[0]["rule"], frames[0]["kind"]))
    for scenario in chosen:
        # Skip a scenario only when every planned row already exists.
        needed = int(scenario["n_replications"]) * 4 * len(kinds)
        # best x non-procrustes are not run, so this overestimates. Resume inside.
        print(f"scenario {scenario['name']} reps={scenario['n_replications']}", flush=True)
        fresh = run_scenario(scenario, out, args.seed0, kinds)
        if frames and len(frames[0]):
            key = ["scenario", "replication", "rule", "kind"]
            existing = frames[0].set_index(key).index
            keep = ~fresh.set_index(key).index.isin(existing)
            fresh = fresh.loc[np.asarray(keep)]
        frames.append(fresh)
        pd.concat(frames, ignore_index=True).to_csv(dest, index=False)
    detail = pd.concat(frames, ignore_index=True)
    detail.to_csv(dest, index=False)
    summary = (
        detail.groupby(["scenario", "rule", "kind"], dropna=False)[
            ["intrinsic_triplet_balanced", "u_triplet", "v_triplet", "n_members"]
        ]
        .agg(["mean", "std", "count"])
    )
    summary.to_csv(out / "summary.csv")
    print(summary.round(3).to_string(), flush=True)
    print(f"wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
