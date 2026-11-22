"""Score multi-view baselines on the Checkpoint 4 product manifolds.

Privileged baselines see the known feature blocks. Isomap on the full
matrix does not. Latent coordinates are used only after an embedding exists.

    PYTHONPATH=redr python redr/scripts/run_baselines.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "redr"))

from redr.baselines import (  # noqa: E402
    aggregated_distance_mds,
    block_distance_embeddings,
    block_matrix,
    concat_one_dimensional,
    isomap_embedding,
    sklearn_mds_aggregated,
)
from redr.complementary import aggregate, score_embedding, triplets_for  # noqa: E402
from redr.intrinsic import triplet_order_disagreement  # noqa: E402
from redr.product import generate_product  # noqa: E402


CONDITIONS = [
    {"name": "square", "aspect": (1.0, 1.0), "hide_second": False, "preserve_scale": False, "seed_offset": 4000},
    {"name": "stretched", "aspect": (5.0, 1.0), "hide_second": False, "preserve_scale": True, "seed_offset": 4300},
    {"name": "square_hidden", "aspect": (1.0, 1.0), "hide_second": True, "preserve_scale": False, "seed_offset": 4200},
]


def _data(spec: dict, replication: int) -> dict:
    return generate_product(
        n_samples=200,
        aspect=spec["aspect"],
        noise=0.05,
        n_nuisance=8,
        hide_second=spec["hide_second"],
        preserve_scale=spec["preserve_scale"],
        random_state=spec["seed_offset"] + replication,
    )


def _record(name: str, privileged: bool, embedding: np.ndarray, data: dict, triplets: np.ndarray) -> dict:
    scores = score_embedding(data, embedding, triplets)
    return {
        "baseline": name,
        "privileged_blocks": privileged,
        "intrinsic_triplet_balanced": scores["intrinsic_triplet_balanced"],
        "intrinsic_triplet_isometric": scores["intrinsic_triplet_isometric"],
        "u_triplet": scores["u_triplet"],
        "v_triplet": scores["v_triplet"],
        "trustworthiness": scores["trustworthiness"],
    }


def _source_balanced(condition: str, replication: int, data: dict, triplets: np.ndarray) -> dict | None:
    path = (
        REPO
        / "results"
        / "checkpoint4"
        / "complementary"
        / condition
        / f"rep{replication:02d}"
    )
    archive_path = path / "embeddings.npz"
    frame_path = path / "candidates.csv"
    if not archive_path.exists() or not frame_path.exists():
        return None
    frame = pd.read_csv(frame_path)
    archive = np.load(archive_path)
    embeddings = {
        str(embedding_id): archive["embeddings"][i]
        for i, embedding_id in enumerate(archive["ids"])
    }
    weighted = frame.copy()
    weighted["method"] = weighted["source"]
    # aggregate() with scheme balanced uses family, so pass uniform and pre-balanced
    # weights by calling the consensus fit directly through a one-row trick:
    members = weighted
    members = members.copy()
    members["family"] = members["source"]
    embedding = aggregate(members, embeddings, "comds", "balanced", path / "baselines", "source_balanced")
    row = _record("c4_pool_source_balanced_comds", True, embedding, data, triplets)
    return row


def metric_audit(data: dict, triplets: np.ndarray) -> dict:
    perfect = score_embedding(data, data["intrinsic"], triplets)
    only_u = score_embedding(
        data, np.column_stack([data["u"], np.zeros(data["n"])]), triplets
    )
    return {
        "disagreement_raw_vs_first_axis": triplet_order_disagreement(data["intrinsic"], triplets),
        "perfect_balanced": perfect["intrinsic_triplet_balanced"],
        "perfect_isometric": perfect["intrinsic_triplet_isometric"],
        "perfect_u": perfect["u_triplet"],
        "perfect_v": perfect["v_triplet"],
        "only_u_balanced": only_u["intrinsic_triplet_balanced"],
        "only_u_u": only_u["u_triplet"],
        "only_u_v": only_u["v_triplet"],
    }


def main() -> None:
    out = REPO / "results" / "checkpoint5" / "baselines"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    audits = []
    for spec in CONDITIONS:
        for replication in range(4):
            data = _data(spec, replication)
            triplets = triplets_for(data["n"], 7000 + replication)
            audit = metric_audit(data, triplets)
            audit.update({"condition": spec["name"], "replication": replication})
            audits.append(audit)
            built = {
                "isomap_full": (False, isomap_embedding(data["X"], n_neighbors=10)),
                "block_standardized_isomap": (True, isomap_embedding(block_matrix(data), n_neighbors=10)),
                "block_standardized_pca": (
                    True,
                    PCA(n_components=2, random_state=0).fit_transform(block_matrix(data)),
                ),
                "concat_1d_isomap": (True, concat_one_dimensional(data)),
                "aggregated_distance_mds": (True, aggregated_distance_mds(data)),
            }
            try:
                built["aggregated_smacof"] = (True, sklearn_mds_aggregated(data))
            except Exception as exc:  # noqa: BLE001
                print(f"smacof failed: {exc}", flush=True)
            views = block_distance_embeddings(data)
            view_map = {f"b{i}": view for i, view in enumerate(views)}
            members = pd.DataFrame({"embedding_id": list(view_map), "family": list(view_map), "method": list(view_map)})
            comds = aggregate(
                members,
                view_map,
                "comds",
                "uniform",
                out / spec["name"] / f"rep{replication:02d}",
                "block_comds",
            )
            built["block_distance_comds"] = (True, comds)
            for name, (privileged, embedding) in built.items():
                row = _record(name, privileged, np.asarray(embedding, dtype=float), data, triplets)
                row.update({"condition": spec["name"], "replication": replication})
                rows.append(row)
                print(
                    f"{spec['name']} r{replication} {name} bal={row['intrinsic_triplet_balanced']:.3f} "
                    f"u={row['u_triplet']:.3f} v={row['v_triplet']:.3f}",
                    flush=True,
                )
            balanced = _source_balanced(spec["name"], replication, data, triplets)
            if balanced is not None:
                balanced.update({"condition": spec["name"], "replication": replication})
                rows.append(balanced)
                print(
                    f"{spec['name']} r{replication} source_balanced bal={balanced['intrinsic_triplet_balanced']:.3f} "
                    f"u={balanced['u_triplet']:.3f} v={balanced['v_triplet']:.3f}",
                    flush=True,
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "detail.csv", index=False)
    summary = frame.groupby(["condition", "baseline", "privileged_blocks"])[
        ["intrinsic_triplet_balanced", "u_triplet", "v_triplet", "trustworthiness", "intrinsic_triplet_isometric"]
    ].agg(["mean", "std"])
    summary.to_csv(out / "summary.csv")
    pd.DataFrame(audits).to_csv(out / "metric_audit.csv", index=False)
    print(summary.round(3).to_string(), flush=True)
    print(pd.DataFrame(audits).groupby("condition").mean(numeric_only=True).round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
