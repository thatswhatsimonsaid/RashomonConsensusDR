"""Run an independent manifold replication.

Examples:

    PYTHONPATH=redr python redr/scripts/run_intrinsic_replication.py \
        --config experiments/study1_s_curve_replication/config.yaml --stage diagnose

    PYTHONPATH=redr python redr/scripts/run_intrinsic_replication.py \
        --config experiments/study1_s_curve_replication/config.yaml --reps 0 --stage all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from redr.replication import (  # noqa: E402
    generate_candidates,
    load_config,
    lock_primary_metric,
    run_consensus,
    summarize,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reps", default="0,1,2,3,4")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["diagnose", "candidates", "consensus", "summarize", "all"],
    )
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO / args.config
    config = load_config(config_path)
    replications = [int(part) for part in args.reps.split(",") if part.strip() != ""]
    lock_path = config["out"] / "geometry_lock.json"
    if args.stage in {"diagnose", "all"}:
        lock_primary_metric(config, replications)
    elif not lock_path.exists():
        raise SystemExit(
            f"Missing {lock_path}. Run --stage diagnose on the full replication "
            "list before candidate or consensus jobs. A single array task must "
            "not choose the primary metric from one dataset."
        )
    if args.stage in {"candidates", "all"}:
        for replication in replications:
            generate_candidates(config, replication)
    if args.stage in {"consensus", "all"}:
        for replication in replications:
            run_consensus(config, replication, workers=args.workers)
    if args.stage in {"summarize", "all"}:
        summarize(config, replications)


if __name__ == "__main__":
    main()
