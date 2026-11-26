"""Run the Checkpoint 7 quality × diversity factorial.

CoMDS and, on a frozen list of cells, LoCoMDS are the aggregators. Procrustes
averaging is a secondary baseline. Oracle selection is labeled and is not a
primary rule.

    PYTHONPATH=redr python redr/scripts/run_quality_diversity.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "redr"))

from redr.comds_bridge import fit_consensus  # noqa: E402
from redr.quality_diversity import (  # noqa: E402
    build_pool,
    cells,
    load_config,
    mean_disagreement,
    procrustes_consensus,
    retained,
    scheme_weights,
    score_consensus,
    size_matched,
    trimmed,
)


def _near(left: float, right: float) -> bool:
    if np.isinf(left) or np.isinf(right):
        return bool(np.isinf(left) and np.isinf(right))
    return abs(float(left) - float(right)) <= 1e-8


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return float("nan")
    value = spearmanr(left, right).correlation
    return float(value) if value == value else float("nan")


def _fit(views: list[np.ndarray], dest: Path, kind: str, weights: np.ndarray | None) -> tuple[np.ndarray, float]:
    dest.mkdir(parents=True, exist_ok=True)
    cached = dest / "gspace.csv"
    if cached.exists() and (dest / "meta.txt").exists():
        embedding = pd.read_csv(cached, header=None).to_numpy(dtype=float)
        seconds = float("nan")
        for line in (dest / "meta.txt").read_text().splitlines():
            if line.startswith("seconds="):
                seconds = float(line.split("=", 1)[1])
        return embedding, seconds
    started = time.perf_counter()
    fitted = fit_consensus(views, dest, kind=kind, weights=weights)
    elapsed = float(fitted.get("seconds", time.perf_counter() - started))
    return np.asarray(fitted["embedding"], dtype=float), elapsed


def _row(cell, replication, rule, epsilon, kind, members, pool, embedding, seconds, spearman) -> dict:
    frame = pool["frame"]
    scores = score_consensus(pool["latent"], embedding, pool["triplets"])
    return {
        "quality": cell["quality"],
        "diversity": cell["diversity"],
        "score_kind": cell["score_kind"],
        "replication": replication,
        "rule": rule,
        "epsilon": epsilon,
        "kind": kind,
        "n_kept": int(len(members)),
        "n_methods_kept": int(members["method"].nunique()) if len(members) else 0,
        "mean_score": float(members["score"].mean()) if len(members) else float("nan"),
        "mean_true_error": float(members["true_global_error"].mean()) if len(members) else float("nan"),
        "retained_disagreement": mean_disagreement(members, pool["embeddings"]),
        "seconds": seconds,
        "score_error_spearman": spearman,
        "pool_mean_error": float(frame["true_global_error"].mean()),
        **scores,
        "error": "",
    }


def _members_for(frame: pd.DataFrame, spec: dict, config: dict, rng: np.random.Generator) -> pd.DataFrame:
    kind = spec["select"]
    if kind == "all":
        return frame.copy()
    if kind == "best":
        return frame.sort_values(["score", "embedding_id"], kind="mergesort").iloc[:1].copy()
    if kind == "trimmed":
        return trimmed(frame, float(config["trim_fraction"]))
    if kind == "rashomon":
        return retained(frame, float(spec["epsilon"]), spec.get("column", "score"))
    if kind == "size":
        return size_matched(frame, int(spec["count"]), rng)
    raise ValueError(kind)


def run_replication(config: dict, cell: dict, replication: int, fit_root: Path) -> pd.DataFrame:
    pool = build_pool(config, cell, replication)
    frame = pool["frame"]
    spearman = _spearman(frame["score"].to_numpy(), frame["true_global_error"].to_numpy())
    rashomon_counts = {eps: len(retained(frame, eps)) for eps in config["epsilons"]}
    locomds = [cell["quality"], cell["diversity"], cell["score_kind"]] in [
        list(item) for item in config["locomds_cells"]
    ]
    specs = [
        {"rule": "best", "select": "best", "epsilon": 0.0, "aggregator": "identity", "scheme": "uniform"},
        {"rule": "unrestricted", "select": "all", "epsilon": np.inf, "aggregator": "comds", "scheme": "uniform"},
        {"rule": "unrestricted_procrustes", "select": "all", "epsilon": np.inf, "aggregator": "procrustes", "scheme": "uniform"},
        {"rule": "trimmed", "select": "trimmed", "epsilon": np.nan, "aggregator": "comds", "scheme": "uniform"},
        {"rule": "full_gibbs", "select": "all", "epsilon": np.inf, "aggregator": "comds", "scheme": "gibbs"},
        {"rule": "full_balanced", "select": "all", "epsilon": np.inf, "aggregator": "comds", "scheme": "balanced"},
    ]
    if locomds:
        specs.append({"rule": "unrestricted_locomds", "select": "all", "epsilon": np.inf, "aggregator": "locomds", "scheme": "uniform"})
    for eps in config["epsilons"]:
        specs.append({"rule": "rashomon_uniform", "select": "rashomon", "epsilon": eps, "aggregator": "comds", "scheme": "uniform"})
        if _near(eps, 0.0) or _near(eps, 0.05):
            specs.append({"rule": "rashomon_balanced", "select": "rashomon", "epsilon": eps, "aggregator": "comds", "scheme": "balanced"})
        if _near(eps, 0.05):
            specs.append({"rule": "rashomon_gibbs", "select": "rashomon", "epsilon": eps, "aggregator": "comds", "scheme": "gibbs"})
            specs.append({"rule": "rashomon_procrustes", "select": "rashomon", "epsilon": eps, "aggregator": "procrustes", "scheme": "uniform"})
            if locomds:
                specs.append({"rule": "rashomon_locomds", "select": "rashomon", "epsilon": eps, "aggregator": "locomds", "scheme": "uniform"})
    if cell["score_kind"] == "informative":
        specs.append(
            {
                "rule": "oracle_rashomon",
                "select": "rashomon",
                "epsilon": 0.05,
                "aggregator": "comds",
                "scheme": "uniform",
                "column": "true_global_error",
            }
        )
    match_count = rashomon_counts[0.05]
    for draw in range(int(config["size_matched_draws"])):
        specs.append(
            {
                "rule": f"size_matched_{draw}",
                "select": "size",
                "epsilon": 0.05,
                "aggregator": "comds",
                "scheme": "uniform",
                "count": match_count,
                "draw": draw,
            }
        )
    rows = []
    cache: dict[tuple, tuple[np.ndarray, float]] = {}
    match_rng = np.random.default_rng(int(config["seed0"]) + 70_000 + replication)
    for spec in specs:
        members = _members_for(frame, spec, config, match_rng)
        ids = tuple(str(value) for value in members["embedding_id"])
        weights = scheme_weights(members, spec["scheme"], float(config["gibbs_beta"]))
        weight_key = None if weights is None else tuple(np.round(weights, 6))
        key = (spec["aggregator"], weight_key, ids)
        try:
            if spec["aggregator"] == "identity":
                embedding = pool["embeddings"][ids[0]]
                seconds = 0.0
            elif spec["aggregator"] == "procrustes":
                embedding = procrustes_consensus(members, pool["embeddings"], weights)
                seconds = 0.0
            elif key in cache:
                embedding, seconds = cache[key]
            else:
                dest = fit_root / spec["rule"] / f"eps{spec['epsilon']}"
                if spec["rule"].startswith("size_matched"):
                    dest = fit_root / spec["rule"]
                views = [pool["embeddings"][item] for item in ids]
                embedding, seconds = _fit(views, dest, "locomds" if spec["aggregator"] == "locomds" else "comds", weights)
                cache[key] = (embedding, seconds)
            row = _row(cell, replication, spec["rule"], spec["epsilon"], spec["aggregator"], members, pool, embedding, seconds, spearman)
        except Exception as exc:  # noqa: BLE001
            row = {
                "quality": cell["quality"],
                "diversity": cell["diversity"],
                "score_kind": cell["score_kind"],
                "replication": replication,
                "rule": spec["rule"],
                "epsilon": spec["epsilon"],
                "kind": spec["aggregator"],
                "n_kept": int(len(members)),
                "n_methods_kept": int(members["method"].nunique()) if len(members) else 0,
                "mean_score": float("nan"),
                "mean_true_error": float("nan"),
                "retained_disagreement": float("nan"),
                "seconds": float("nan"),
                "score_error_spearman": spearman,
                "pool_mean_error": float(frame["true_global_error"].mean()),
                "intrinsic_triplet_balanced": float("nan"),
                "intrinsic_triplet_isometric": float("nan"),
                "arc_triplet_accuracy": float("nan"),
                "height_triplet_accuracy": float("nan"),
                "intrinsic_rank_correlation": float("nan"),
                "error": f"{type(exc).__name__}: {exc}",
            }
        rows.append(row)
        print(
            f"{cell['quality']}/{cell['diversity']}/{cell['score_kind']} r{replication} "
            f"{spec['rule']} eps={spec['epsilon']} n={row['n_kept']} "
            f"balanced={row['intrinsic_triplet_balanced'] if row['intrinsic_triplet_balanced'] == row['intrinsic_triplet_balanced'] else float('nan'):.3f} "
            f"{row['error'][:80]}",
            flush=True,
        )
    return pd.DataFrame(rows)


def main() -> None:
    config = load_config()
    root = REPO / "results" / "checkpoint7" / "quality_diversity"
    root.mkdir(parents=True, exist_ok=True)
    written = []
    for cell in cells(config):
        for replication in range(int(config["replications"])):
            name = f"{cell['quality']}__{cell['diversity']}__{cell['score_kind']}__rep{replication}.csv"
            dest = root / "by_rep" / name
            if dest.exists():
                written.append(dest)
                print(f"cached {name}", flush=True)
                continue
            fit_root = root / "fits" / name.replace(".csv", "")
            frame = run_replication(config, cell, replication, fit_root)
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
