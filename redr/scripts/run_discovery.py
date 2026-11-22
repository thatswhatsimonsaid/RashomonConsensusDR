"""Local pilot: can an unsupervised rule find complementary views?

    PYTHONPATH=redr python redr/scripts/run_discovery.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "redr"))

from redr.baselines import (  # noqa: E402
    aggregated_distance_mds,
    block_distance_embeddings,
    concat_one_dimensional,
    isomap_embedding,
)
from redr.checkpoint2 import _fit_or_load  # noqa: E402
from redr.complementary import score_embedding, triplets_for  # noqa: E402
from redr.controlled import weighted_mean  # noqa: E402
from redr.discovery import (  # noqa: E402
    OBJECTIVE_COLUMNS,
    discover_candidates,
    make_dataset,
    marginal_select,
    quality_weights,
    score_objectives,
)
from redr.replication import _content_scheme  # noqa: E402
from redr.selection_rules import diversity_first, pareto_front, pooled, union_of_objectives  # noqa: E402


N_SAMPLES = 160
N_REPS = 3
EPSILON = 0.05

CONDITIONS = [
    {"name": "aligned", "mixing": "aligned", "hide_second": False, "noise": 0.05, "n_nuisance": 8, "preserve_scale": False, "aspect": (1.0, 1.0), "contaminate": False, "seed_offset": 6000},
    {"name": "rotated", "mixing": "rotated", "hide_second": False, "noise": 0.05, "n_nuisance": 8, "preserve_scale": False, "aspect": (1.0, 1.0), "contaminate": False, "seed_offset": 6100},
    {"name": "rotated_hidden", "mixing": "rotated", "hide_second": True, "noise": 0.05, "n_nuisance": 8, "preserve_scale": False, "aspect": (1.0, 1.0), "contaminate": False, "seed_offset": 6200},
    {"name": "rotated_contaminated", "mixing": "rotated", "hide_second": False, "noise": 0.05, "n_nuisance": 8, "preserve_scale": False, "aspect": (1.0, 1.0), "contaminate": True, "seed_offset": 6100},
    {"name": "stretched_rotated", "mixing": "rotated", "hide_second": False, "noise": 0.05, "n_nuisance": 8, "preserve_scale": True, "aspect": (5.0, 1.0), "contaminate": False, "seed_offset": 6300},
    {"name": "rotated_nuisance20", "mixing": "rotated", "hide_second": False, "noise": 0.05, "n_nuisance": 20, "preserve_scale": False, "aspect": (1.0, 1.0), "contaminate": False, "seed_offset": 6400},
    {"name": "rotated_noise025", "mixing": "rotated", "hide_second": False, "noise": 0.25, "n_nuisance": 8, "preserve_scale": False, "aspect": (1.0, 1.0), "contaminate": False, "seed_offset": 6500},
]


def _fit(members: pd.DataFrame, embeddings: dict, kind: str, out: Path, name: str, weights: np.ndarray | None = None) -> np.ndarray:
    ids = [str(value) for value in members["embedding_id"]]
    views = [np.asarray(embeddings[key], dtype=float) for key in ids]
    if weights is None:
        weights = np.ones(len(views), dtype=float)
    if len(views) == 1:
        return views[0]
    if kind == "procrustes":
        return weighted_mean(views, weights)
    fitted = _fit_or_load(out, kind, views, weights, None, ids, _content_scheme(f"{name}|{kind}", views))
    return np.asarray(fitted["embedding"], dtype=float)


def _row(name: str, privileged: bool, embedding: np.ndarray, data: dict, triplets: np.ndarray, n_members: int) -> dict:
    scores = score_embedding(data, embedding, triplets)
    return {
        "rule": name,
        "privileged_blocks": privileged,
        "n_members": n_members,
        "intrinsic_triplet_balanced": scores["intrinsic_triplet_balanced"],
        "u_triplet": scores["u_triplet"],
        "v_triplet": scores["v_triplet"],
        "trustworthiness": scores["trustworthiness"],
        "error": "",
    }


def run_one(spec: dict, replication: int, root: Path) -> list[dict]:
    data = make_dataset(
        n_samples=N_SAMPLES,
        mixing=spec["mixing"],
        hide_second=spec["hide_second"],
        noise=spec["noise"],
        n_nuisance=spec["n_nuisance"],
        preserve_scale=spec["preserve_scale"],
        aspect=spec["aspect"],
        random_state=spec["seed_offset"] + replication,
        contaminate=spec["contaminate"],
    )
    triplets = triplets_for(data["n"], 12000 + replication)
    embeddings, subsets = discover_candidates(
        data["X_visible"], data["random_state"] + 3, contaminate=spec["contaminate"]
    )
    frame = score_objectives(data["X_visible"], embeddings, subsets)
    out = root / spec["name"] / f"rep{replication:02d}"
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "objectives.csv", index=False)
    rows = []

    def evaluate(name: str, members: pd.DataFrame, kind: str, privileged: bool, weights=None) -> None:
        try:
            if len(members) == 0:
                raise RuntimeError("empty selection")
            embedding = _fit(members, embeddings, kind, out, name, weights=weights)
            row = _row(name, privileged, embedding, data, triplets, len(members))
        except Exception as exc:  # noqa: BLE001
            row = {
                "rule": name,
                "privileged_blocks": privileged,
                "n_members": int(len(members)),
                "error": f"{type(exc).__name__}: {exc}",
            }
        row.update({"condition": spec["name"], "replication": replication})
        rows.append(row)
        print(
            f"{spec['name']} r{replication} {name} n={row['n_members']} "
            f"bal={row.get('intrinsic_triplet_balanced', float('nan'))} "
            f"u={row.get('u_triplet', float('nan'))} v={row.get('v_triplet', float('nan'))} "
            f"err={row['error'][:60]}",
            flush=True,
        )

    evaluate("all_comds", frame, "comds", False)
    evaluate("all_procrustes", frame, "procrustes", False)
    evaluate("pooled_multiscale_comds", pooled(frame, EPSILON, loss_col="multiscale_loss"), "comds", False)
    evaluate(
        "union_comds",
        union_of_objectives(frame, list(OBJECTIVE_COLUMNS), EPSILON),
        "comds",
        False,
    )
    evaluate("pareto_comds", pareto_front(frame, list(OBJECTIVE_COLUMNS)), "comds", False)
    evaluate(
        "marginal_comds",
        marginal_select(frame, embeddings, data["X_visible"], subsets),
        "comds",
        False,
    )
    evaluate("diversity_first_comds", diversity_first(frame, embeddings, k=10, keep_max=6), "comds", False)
    weights = quality_weights(frame)
    evaluate("quality_weighted_comds", frame, "comds", False, weights=weights)
    best = frame.loc[[frame["multiscale_loss"].idxmin()]]
    evaluate("best_multiscale", best, "procrustes", False)

    # Privileged integrations. They see the original blocks, not the rotated matrix.
    privileged = {
        "privileged_concat_1d": concat_one_dimensional(data),
        "privileged_distance_mds": aggregated_distance_mds(data),
        "privileged_isomap_visible": isomap_embedding(data["X_visible"], n_neighbors=10),
    }
    for name, embedding in privileged.items():
        row = _row(name, name != "privileged_isomap_visible", embedding, data, triplets, 1)
        row.update({"condition": spec["name"], "replication": replication})
        rows.append(row)
        print(
            f"{spec['name']} r{replication} {name} bal={row['intrinsic_triplet_balanced']:.3f} "
            f"u={row['u_triplet']:.3f} v={row['v_triplet']:.3f}",
            flush=True,
        )
    views = block_distance_embeddings(data)
    view_map = {f"block_{i}": view for i, view in enumerate(views)}
    members = pd.DataFrame({"embedding_id": list(view_map)})
    try:
        embedding = _fit(members, view_map, "comds", out, "privileged_block_comds")
        row = _row("privileged_block_comds", True, embedding, data, triplets, len(members))
    except Exception as exc:  # noqa: BLE001
        row = {"rule": "privileged_block_comds", "privileged_blocks": True, "n_members": 2, "error": str(exc)}
    row.update({"condition": spec["name"], "replication": replication, "error": row.get("error", "")})
    rows.append(row)
    print(
        f"{spec['name']} r{replication} privileged_block_comds "
        f"bal={row.get('intrinsic_triplet_balanced', float('nan'))}",
        flush=True,
    )
    return rows


def main() -> None:
    root = REPO / "results" / "checkpoint5" / "discovery"
    root.mkdir(parents=True, exist_ok=True)
    dest = root / "detail.csv"
    done = set()
    frames = []
    if dest.exists():
        previous = pd.read_csv(dest)
        frames.append(previous)
        done = set(zip(previous["condition"], previous["replication"], previous["rule"]))
    for spec in CONDITIONS:
        for replication in range(N_REPS):
            if (spec["name"], replication, "all_comds") in done:
                print(f"skip {spec['name']} {replication}", flush=True)
                continue
            fresh = pd.DataFrame(run_one(spec, replication, root))
            frames.append(fresh)
            pd.concat(frames, ignore_index=True).to_csv(dest, index=False)
    detail = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    detail.to_csv(dest, index=False)
    if len(detail) and "intrinsic_triplet_balanced" in detail.columns:
        summary = detail.groupby(["condition", "rule"])[
            ["intrinsic_triplet_balanced", "u_triplet", "v_triplet", "trustworthiness", "n_members"]
        ].agg(["mean", "std", "count"])
        summary.to_csv(root / "summary.csv")
        print(summary.round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
