"""Fresh seeds for the five cells that carried the Checkpoint 7 mechanism.

The rules, epsilon, metrics, and generators are unchanged. Only seed0 changes,
from 11000 to 21000. LoCoMDS is omitted here; the pilot already ran it.

    PYTHONPATH=redr python redr/scripts/run_quality_diversity_confirm.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "redr"))

CELLS = (
    {"quality": "severe", "diversity": "identical", "score_kind": "informative"},
    {"quality": "high", "diversity": "identical", "score_kind": "informative"},
    {"quality": "high", "diversity": "harmful", "score_kind": "informative"},
    {"quality": "mixed", "diversity": "complementary", "score_kind": "informative"},
    {"quality": "mixed", "diversity": "complementary", "score_kind": "misleading"},
)


def main() -> None:
    spec = importlib.util.spec_from_file_location("runqd", REPO / "redr" / "scripts" / "run_quality_diversity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.load_config()
    config["seed0"] = 21000
    config["locomds_cells"] = []
    root = REPO / "results" / "checkpoint7" / "confirm"
    written = []
    for cell in CELLS:
        for replication in range(int(config["replications"])):
            name = f"{cell['quality']}__{cell['diversity']}__{cell['score_kind']}__rep{replication}.csv"
            dest = root / "by_rep" / name
            if dest.exists():
                written.append(dest)
                continue
            frame = module.run_replication(config, cell, replication, root / "fits" / name.replace(".csv", ""))
            dest.parent.mkdir(parents=True, exist_ok=True)
            temporary = dest.with_suffix(".tmp")
            frame.to_csv(temporary, index=False)
            temporary.replace(dest)
            written.append(dest)
    detail = pd.concat([pd.read_csv(path) for path in written], ignore_index=True)
    detail.to_csv(root / "detail.csv", index=False)
    print(f"wrote {root / 'detail.csv'} rows={len(detail)}", flush=True)


if __name__ == "__main__":
    main()
